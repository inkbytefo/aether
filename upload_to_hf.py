import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_models(repo_id, token=None):
    print(f"🚀 Starting backup to Hugging Face Hub: {repo_id}")
    
    api = HfApi(token=token)
    
    # 1. Create Repo (if not exists)
    try:
        create_repo(repo_id, private=True, exist_ok=True, token=token)
        print(f"✅ Repository ensured: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation warning (might already exist): {e}")

    # 2. Upload Models Directory
    models_dir = "models/saved"
    if not os.path.exists(models_dir):
        print(f"❌ Directory {models_dir} not found!")
        return

    print(f"📤 Uploading {models_dir}...")
    try:
        api.upload_folder(
            folder_path=models_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="models/saved",
            token=token
        )
        print("✅ Upload complete! Your models are safe.")
        
        # Also upload config and tokenizer for reproducibility
        print("📤 Uploading configs and tokenizer...")
        api.upload_folder(
            folder_path="configs",
            repo_id=repo_id,
            path_in_repo="configs",
            token=token
        )
        if os.path.exists("data/corpus_v1/tokenizer.model"):
            api.upload_file(
                path_or_fileobj="data/corpus_v1/tokenizer.model",
                path_in_repo="tokenizer.model",
                repo_id=repo_id,
                token=token
            )
            
        print("🎉 All backups finished successfully.")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace Repo ID (e.g., username/aether-backup)")
    parser.add_argument("--token", type=str, help="HF Write Token (optional if logged in via CLI)")
    args = parser.parse_args()
    
    upload_models(args.repo, args.token)
