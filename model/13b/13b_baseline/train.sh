#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"

sh "$DIR/../../prepare.sh"

nohup python3 "$DIR/llama-pubmed-211k.py" \
	--report_to wandb \
	--run_name llama-2-13b-pubmed-qa \
	&
