import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import argparse

from src.utils.config import Config
from src.models.mamba import MambaLLM
from src.data.dataset import create_dataloaders

def train(config_path: str, resume_from: str = None):
    # Load Config
    cfg = Config.load(config_path)
    
    # Setup Device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize WandB
    wandb.init(project="AETHER-1", config=cfg.__dict__, name=f"phase2-{os.path.basename(config_path)}")

    # Create DataLoaders
    print("Loading data...")
    # Pass the full config object to create_dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(cfg)
    
    # CRITICAL: Sync vocab size
    if len(tokenizer) != cfg.model.vocab_size:
        print(f"⚠️ Config vocab_size ({cfg.model.vocab_size}) does not match Tokenizer ({len(tokenizer)}). Updating config.")
        cfg.model.vocab_size = len(tokenizer)

    # Initialize Model
    print("Initializing model...")
    if cfg.model.use_plasticity:
        from src.models.plastic_mamba import PlasticMambaLLM
        print("🧠 Using PlasticMambaLLM with Hebbian Memory")
        model = PlasticMambaLLM(cfg.model).to(device)
    else:
        model = MambaLLM(cfg.model).to(device)
    
    # Load Checkpoint if provided
    if resume_from:
        print(f"Loading checkpoint from {resume_from}...")
        if os.path.exists(resume_from):
            state_dict = torch.load(resume_from, map_location=device)
            
            # Handle partial loading for PlasticMambaLLM
            if cfg.model.use_plasticity:
                print("⚠️ Loading partial weights for Plastic Architecture...")
                model_dict = model.state_dict()
                # Filter out unnecessary keys and match shapes if possible
                # MambaLMHeadModel keys might differ slightly from our custom PlasticMambaLLM
                # Our PlasticMambaLLM uses 'backbone' implicitly via layers? No, we defined self.layers
                # MambaLMHeadModel has 'backbone.layers...'
                
                # Let's try to map keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Map 'backbone.layers.X' to 'layers.X.mixer'
                    if k.startswith("backbone.layers"):
                        # k is like backbone.layers.0.mixer...
                        # We need to map it to layers.0.mixer...
                        new_k = k.replace("backbone.layers", "layers")
                        # Also MambaLMHeadModel blocks might have different internal names
                        # Let's just try strict=False loading with what matches
                        if new_k in model_dict:
                            new_state_dict[new_k] = v
                    elif k.startswith("backbone.embedding"):
                        new_state_dict["embedding.weight"] = v
                    elif k.startswith("backbone.norm_f"):
                        new_state_dict["norm_f.weight"] = v
                    elif k.startswith("lm_head"):
                        new_state_dict[k] = v
                
                # Load with strict=False
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded keys: {len(new_state_dict)}")
                print(f"Missing keys (expected for new Hebbian layers): {len(missing)}")
            else:
                model.load_state_dict(state_dict)
                print("✅ Checkpoint loaded.")
        else:
            print(f"⚠️ Checkpoint {resume_from} not found. Starting from scratch.")
    
    # Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training Loop
    model.train()
    step = 0
    micro_step = 0
    pbar = tqdm(total=cfg.training.max_steps, desc="Training")
    
    while step < cfg.training.max_steps:
        for batch in train_loader:
            if step >= cfg.training.max_steps:
                break
                
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Calculate Loss
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                main_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Auxiliary Loss: Z-Loss
                z_loss_weight = 2e-4
                log_z = torch.logsumexp(logits, dim=-1)
                aux_loss = z_loss_weight * (log_z ** 2).mean()
                
                loss = main_loss + aux_loss
                
                # Gradient Accumulation
                loss = loss / cfg.training.gradient_accumulation_steps
            
            # Backward pass with Scaler
            scaler.scale(loss).backward()
            
            # Optimizer Step (only every N steps)
            if (micro_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                # Unscale and Clip Grads
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Logging
                wandb.log({"train_loss": main_loss.item(), "aux_loss": getattr(aux_loss, 'item', lambda: 0)(), "step": step})
                pbar.set_postfix({"loss": f"{main_loss.item():.4f}"})
                pbar.update(1)
                step += 1
                
                # Validation (every 100 steps)
                if step % 100 == 0:
                    validate(model, val_loader, device, step)
                    model.train()
            
            micro_step += 1

    # Save Model
    os.makedirs("models/saved", exist_ok=True)
    
    # Determine save name based on config path
    config_name = os.path.basename(config_path)
    if "phase1" in config_name:
        save_name = "aether_phase1.pt"
    elif "phase2" in config_name:
        save_name = "aether_phase2.pt"
    elif "phase3" in config_name:
        save_name = "aether_phase3.pt"
    elif "phase4" in config_name:
        save_name = "aether_phase4.pt"
    else:
        save_name = "aether_model.pt"
        
    torch.save(model.state_dict(), f"models/saved/{save_name}")
    print(f"Training complete. Model saved to models/saved/{save_name}")

def validate(model, val_loader, device, step):
    model.eval()
    total_loss = 0
    steps = 0
    max_val_steps = 10 # Limit validation steps for speed
    
    with torch.no_grad():
        for batch in val_loader:
            if steps >= max_val_steps:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            steps += 1
            
    avg_loss = total_loss / steps if steps > 0 else 0
    wandb.log({"val_loss": avg_loss, "step": step})
    print(f" [Validation] Step {step}: Loss {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args.config, args.resume_from)
