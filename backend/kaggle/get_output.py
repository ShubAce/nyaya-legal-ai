from huggingface_hub import HfApi

api = HfApi()

info = api.repo_info(
    repo_id="mrroyaleace/nyaya7b-finetuning-output",
    repo_type="dataset"
)

print(info)