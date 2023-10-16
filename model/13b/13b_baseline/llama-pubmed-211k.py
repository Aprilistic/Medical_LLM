import os
from dotenv import load_dotenv

load_dotenv()

import huggingface_hub
import wandb

huggingface_hub.login(token=os.getenv('HF_API_KEY'))

os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
os.environ["WANDB_PROJECT"] = "AIDoc" # log to your project 
os.environ["WANDB_LOG_MODEL"] = "all" # log your models

wandb.init()

import torch
from random import randrange
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

# Load dataset from the hub
dataset = load_dataset("pubmed_qa", "pqa_artificial", split="train")

# dataset.select(range(10000))

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])

def format_instruction(sample):
	return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the response using an LLM.

### Input:
{sample['question']}

### Response:
{sample['long_answer']}
"""

from random import randrange

print(format_instruction(dataset[randrange(len(dataset))]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

use_flash_attention = False

# Hugging Face model id
model_id = "meta-llama/Llama-2-13b-chat-hf"


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map={"":0})
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
    # report_to="wandb",
    output_dir="llama-2-13b-pubmed-qa-211k",
    num_train_epochs=2,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True,  # disable tqdm since with packing values are in correct
)

wandb.config.update(args)



from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)


# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()


import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


args.output_dir = "llama-2-13b-pubmed-qa-211k"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

from datasets import load_dataset
from random import randrange


# Load dataset from the hub and get a sample
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
sample = dataset[randrange(len(dataset))]

prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the response using an LLM.

### Input:
{sample['question']}

### Response:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
with torch.inference_mode():
	outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Prompt:\n{sample['question']}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"Ground truth:\n{sample['long_answer']}")

from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")

# push merged model to the hub
merged_model.push_to_hub("llama-2-13b-pubmed-qa-211k")
tokenizer.push_to_hub("llama-2-13b-pubmed-qa-211k")