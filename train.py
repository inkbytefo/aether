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

def train(config_path: str):
    # Load Config
    cfg = Config.load(config_path)
    
    # Setup Device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize WandB
    wandb.init(project="AETHER-1", config=cfg.__dict__)

    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length
    )

    # Initialize Model
    print("Initializing model...")
    model = MambaLLM(cfg.model).to(device)
    
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
    torch.save(model.state_dict(), "models/saved/aether_phase1.pt")
    print("Training complete. Model saved.")

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
            
    avg_loss = total_loss / steps
    wandb.log({"val_loss": avg_loss, "step": step})
    print(f" [Validation] Step {step}: Loss {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    train(args.config)
