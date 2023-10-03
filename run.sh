#!/bin/sh

# Prepare the environment for the build
pip install --upgrade pip
pip install -r requirements.txt

nohup python3 model/llama-2-7b-pubmed-qa-10000.py \
	--report_to wandb \
	--run_name llama-2-7b-pubmed-qa-10000 \
	&

