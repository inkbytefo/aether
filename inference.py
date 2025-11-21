import torch
import argparse
from src.utils.config import Config
from src.data.tokenizer import Tokenizer
from src.models.plastic_mamba import PlasticMambaLLM
from src.models.mamba import MambaLLM
import os

def generate(model, tokenizer, prompt, max_length=200, temperature=0.7, top_k=40, device="cuda"):
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    batch_size = input_ids.shape[0]
    
    # Prepare inference params for stateful generation
    # This dictionary will hold:
    # 1. SSM State (managed by Mamba mixer)
    # 2. Hebbian State (managed by PlasticMambaBlock)
    inference_params = {
        "max_seqlen": max_length + input_ids.shape[1],
        "max_batch_size": batch_size,
        "seqlen_offset": 0
    }
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        # First pass: Process the prompt
        # The model will process the full prompt and initialize/update states in inference_params
        outputs = model(input_ids, inference_params=inference_params)
        logits = outputs.logits[:, -1, :] / temperature
        
        # Sample first new token
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Update offset
        inference_params["seqlen_offset"] += input_ids.shape[1]
        
        # Generation Loop
        for _ in range(max_length - 1):
            # Feed ONLY the last token
            # This is O(1) per step instead of O(L)
            outputs = model(next_token, inference_params=inference_params)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            inference_params["seqlen_offset"] += 1
            
            # Stop if EOS (optional)
            # if next_token.item() == tokenizer.eos_token_id:
            #     break
                
    return tokenizer.decode(generated_ids[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase4.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/saved/aether_phase4.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode")
    args = parser.parse_args()
    
    # Load Config
    cfg = Config.load(args.config)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Tokenizer
    tokenizer = Tokenizer(max_length=cfg.data.max_length)
    
    # Initialize Model
    print("Initializing model...")
    if cfg.model.use_plasticity:
        print("🧠 Using PlasticMambaLLM")
        model = PlasticMambaLLM(cfg.model).to(device)
    else:
        model = MambaLLM(cfg.model).to(device)
        
    # Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Checkpoint loaded.")
    else:
        print(f"⚠️ Checkpoint {args.checkpoint} not found! Using random weights.")

    # Interactive Loop
    if args.prompt:
        response = generate(model, tokenizer, args.prompt, device=device)
        print(f"\n🤖 AETHER: {response}")
    else:
        print("\n💬 AETHER Inference Mode (Type 'exit' to quit)")
        print("---------------------------------------------")
        while True:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Format prompt to encourage reasoning if not present
            if "Question:" not in user_input:
                prompt = f"Question: {user_input} Answer: <think>"
            else:
                prompt = user_input
                
            print("Thinking...", end="\r")
            response = generate(model, tokenizer, prompt, device=device)
            
            # Clean up response for display
            # If we added <think>, the model continues from there.
            # We want to show the full generation.
            print(f"\n🤖 AETHER:\n{response}")

if __name__ == "__main__":
    main()
