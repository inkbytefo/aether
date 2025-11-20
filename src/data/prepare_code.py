import os
from datasets import load_dataset
import json
from tqdm import tqdm

def prepare_mbpp(output_dir="data/processed/code"):
    print("Downloading MBPP dataset...")
    dataset = load_dataset("mbpp", split="train")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mbpp_train.jsonl")
    
    print(f"Processing {len(dataset)} examples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            # Format: Docstring (Instruction) + Code
            text = f'\"\"\"\n{item["text"]}\n\"\"\"\n{item["code"]}'
            
            entry = {
                "text": text,
                "source": "mbpp",
                "task_id": item["task_id"]
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    prepare_mbpp()
