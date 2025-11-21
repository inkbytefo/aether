import json
import os
import random

def generate_arithmetic_chain():
    """
    Generates simple arithmetic problems with chain-of-thought steps.
    Format: Question -> <think> Step 1... Step 2... </think> -> Answer
    """
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    op = random.choice(['+', '-'])
    
    if op == '+':
        res = a + b
        thought = f"I need to add {a} and {b}. First, add the ones: {a%10} + {b%10} = {(a%10 + b%10)}. Then add the tens: {a//10}0 + {b//10}0 = {(a//10 + b//10)*10}. Finally, combine them."
    else:
        # Ensure positive result for simplicity
        if a < b: a, b = b, a
        res = a - b
        thought = f"I need to subtract {b} from {a}. First subtract the ones, then the tens."

    text = f"Question: What is {a} {op} {b}? Answer: <think> {thought} </think> {res}"
    return {"text": text}

def generate_logic_chain():
    """
    Generates simple logic chains.
    """
    subjects = ["Socrates", "Plato", "Aristotle", "A cat", "A dog"]
    is_human = [True, True, True, False, False]
    idx = random.randint(0, len(subjects)-1)
    
    subj = subjects[idx]
    human = is_human[idx]
    
    if human:
        thought = f"{subj} is a human. All humans are mortal. Therefore {subj} is mortal."
        res = "Yes"
    else:
        thought = f"{subj} is not a human. The premise only applies to humans."
        res = "No"
        
    text = f"Question: Is {subj} mortal? Answer: <think> {thought} </think> {res}"
    return {"text": text}

def main():
    output_dir = "data/processed/reasoning"
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    # Generate 5000 samples
    for _ in range(2500):
        data.append(generate_arithmetic_chain())
        data.append(generate_logic_chain())
        
    random.shuffle(data)
    
    output_file = os.path.join(output_dir, "synthetic_reasoning.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(data)} reasoning samples in {output_file}")

if __name__ == "__main__":
    main()
