#!/bin/sh

# Prepare the environment for the build
pip install -r requirements.txt

source .env

WANDB_API_KEY=$WANDB_API_KEY
WANDB_PROJECT=AIDoc
WANDB_WATCH=all

nohup python3 model/llama-2-7b-pubmed-qa-10000.py \
	--report_to wandb \
	--run_name llama-2-7b-pubmed-qa-10000 \
	&

