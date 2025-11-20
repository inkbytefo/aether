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

    # Initialize Model
    print("Initializing model...")
    model = MambaLLM(cfg.model).to(device)
    
    # Load Checkpoint if provided
    if resume_from:
        print(f"Loading checkpoint from {resume_from}...")
        if os.path.exists(resume_from):
            state_dict = torch.load(resume_from, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Checkpoint loaded.")
        else:
            print(f"⚠️ Checkpoint {resume_from} not found. Starting from scratch.")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training Loop
    model.train()
    step = 0
    pbar = tqdm(total=cfg.training.max_steps, desc="Training")
    
    while step < cfg.training.max_steps:
        for batch in train_loader:
            if step >= cfg.training.max_steps:
                break
                
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Calculate Loss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            wandb.log({"train_loss": loss.item(), "step": step})
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)
            step += 1
            
            # Validation (every 100 steps)
            if step % 100 == 0:
                validate(model, val_loader, device, step)
                model.train()

    # Save Model
    os.makedirs("models/saved", exist_ok=True)
    save_name = "aether_phase2.pt" if "phase2" in config_path else "aether_phase1.pt"
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
