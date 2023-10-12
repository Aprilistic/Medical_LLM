#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"

sh "$DIR/../../prepare.sh"

nohup python3 "$DIR/model/llama-2-7b-patient.py" \
	--report_to wandb \
	--run_name llama-2-7b-pubmed-qa \
	&
