## Developer: inkbytefo
## Modified: 2025-11-21

import torch
import argparse
from src.utils.config import Config
from src.data.tokenizer import Tokenizer
from src.models.plastic_mamba import PlasticMambaLLM
from src.models.mamba import MambaLLM
from mamba_ssm.utils.generation import InferenceParams
import os

def generate(model, tokenizer, prompt, max_length=500, temperature=0.7, top_k=40, device="cuda", debug=False):
    model.eval()
    
    # Encode prompt - tokenizer returns dict with 'input_ids' and 'attention_mask'
    encodings = tokenizer.encode(prompt, return_tensors=True, add_special_tokens=True)
    input_ids = encodings['input_ids'].to(device)
    batch_size = input_ids.shape[0]
    
    if debug:
        print(f"\n[DEBUG] Input length: {input_ids.shape[1]} tokens")
        print(f"[DEBUG] Will generate up to {max_length} new tokens")
    
    # Prepare inference params for stateful generation
    # Use Mamba's InferenceParams class instead of dict
    inference_params = InferenceParams(
        max_seqlen=max_length + input_ids.shape[1],
        max_batch_size=batch_size
    )
    
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
        
        # Update offset for subsequent passes
        inference_params.seqlen_offset += input_ids.shape[1]
        
        # Generation Loop
        for step in range(max_length - 1):
            # Feed ONLY the last token
            # This is O(1) per step instead of O(L)
            outputs = model(next_token, inference_params=inference_params)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if debug and step < 5:
                print(f"[DEBUG] Step {step}: Generated token {next_token.item()}")
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            inference_params.seqlen_offset += 1
            
            # Stop if EOS (optional)
            if next_token.item() == tokenizer.eos_token_id:
                if debug:
                    print(f"[DEBUG] EOS token hit at step {step}")
                break
                
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(model_path=tokenizer_path, max_length=cfg.data.max_length)
    else:
        print("⚠️ Custom tokenizer not found, using fallback (this may not work correctly)")
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
            
            # Use input directly without formatting
            prompt = user_input
                
            print("Thinking...", end="\r")
            response = generate(model, tokenizer, prompt, device=device, debug=True)
            
            print(f"\n🤖 AETHER:\n{response}")

if __name__ == "__main__":
    main()
