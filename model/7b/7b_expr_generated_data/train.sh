#!/bin/sh

sh ../../prepare.sh

nohup python3 model/llama-2-7b-patient.py \
	--report_to wandb \
	--run_name llama-2-7b-pubmed-qa \
	&
