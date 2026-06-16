# run this as a script: push_model.py
from huggingface_hub import HfApi
import os

HF_TOKEN    = "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"
HF_USERNAME = "MrRoyaleAce"
DATASET_ID  = "MrRoyaleAce/nyaya7b-finetuning-output"
MODEL_ID    = "MrRoyaleAce/nyaya-7b"

api = HfApi()

# Create the model repo
api.create_repo(
    repo_id=MODEL_ID,
    repo_type="model",
    private=False,      # public — this is your CV link
    token=HF_TOKEN,
    exist_ok=True,
)
print(f"✓ Model repo created: https://huggingface.co/{MODEL_ID}")

# Copy files from dataset to model repo
# The merged model files are at nyaya-checkpoints/nyaya-7b-merged/ in your dataset
files_to_copy = [
    "nyaya-checkpoints/nyaya-7b-merged/config.json",
    "nyaya-checkpoints/nyaya-7b-merged/model.safetensors",
    "nyaya-checkpoints/nyaya-7b-merged/tokenizer.json",
    "nyaya-checkpoints/nyaya-7b-merged/tokenizer_config.json",
    "nyaya-checkpoints/nyaya-7b-merged/generation_config.json",
    "nyaya-checkpoints/nyaya-7b-merged/chat_template.jinja",
]

for dataset_path in files_to_copy:
    filename = os.path.basename(dataset_path)
    print(f"Copying {filename}...")
    try:
        # Download from dataset, upload to model
        local = api.hf_hub_download(
            repo_id=DATASET_ID,
            filename=dataset_path,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=filename,
            repo_id=MODEL_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename}: {e}")

print(f"\n✓ Model live at: https://huggingface.co/{MODEL_ID}")
print("Add this to your CV!")