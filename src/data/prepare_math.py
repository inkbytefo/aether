import os
from datasets import load_dataset
import json
from tqdm import tqdm

def prepare_gsm8k(output_dir="data/processed/math"):
    print("Downloading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gsm8k_train.jsonl")
    
    print(f"Processing {len(dataset)} examples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            # Format: Question + Answer (Reasoning)
            # We might want to add a special separator or format later
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            
            entry = {
                "text": text,
                "source": "gsm8k",
                "question": item['question'],
                "answer": item['answer']
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    prepare_gsm8k()
