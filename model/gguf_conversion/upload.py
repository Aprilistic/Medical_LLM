from huggingface_hub import HfApi
api = HfApi()

model_id = "BLACKBUN/llama-2-7b-pubmed-qa-211k-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj="llama-2-7b-pubmed-qa-211k.gguf",
    path_in_repo="llama-2-7b-pubmed-qa-211k.gguf",
    repo_id=model_id,
)