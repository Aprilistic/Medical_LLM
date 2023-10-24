import os

import huggingface_hub
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

huggingface_hub.login(token=os.getenv('HF_API_KEY'))

output_dir = "llama-2-7b-knowledge"

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# push merged model to the hub
merged_model.push_to_hub("llama-2-7b-knowledge")
tokenizer.push_to_hub("llama-2-7b-knowledge")