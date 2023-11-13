
# A Step-by-Step Guide

## Environment Setup

Tested on Linux, macOS

### Install prerequisites

1. Install the required packages by executing `prepare.sh`
2. If some packages are not compatible, modify the `requirements.txt` 

```bash
sh ./prepare.sh
```

## Modify `.env` file

1. Huggingface API key
    1. Since we are using huggingface library, you have to set your HF API key. Our python script will upload the model to HF, so you need to generate **WRITE** key.
2. wandb API key (optional)
    1. This is optional but using wandb may help you check progress training your model.

Be careful of uploading your credentials!!

## Train the model
[training_guide.md](https://github.com/Aprilistic/Medical_LLM/blob/main/model/training_guide.ipynb)


## Inference
