from huggingface_hub import snapshot_download
model_id="BLACKBUN/llama-2-7b-pubmed-qa-211k"
snapshot_download(repo_id=model_id, local_dir="llama-2-pubmed",
                  local_dir_use_symlinks=False, revision="main")