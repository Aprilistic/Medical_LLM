# Inference Guide

We used [llama.cpp](https://github.com/ggerganov/llama.cpp) library. You can find detailed usage there.


## Convert the fine-tuned model
Since the original fine-tuned models are heavy, we quantized our models to 8-bit integers. 
1. Download the model and include `tokenizer.model` file from [link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

2. And run the convert script.
  For example
  ```bash
  python3 llama.cpp/convert.py llama-2-both \  
  --outfile llama-2-7b-both.gguf \
  --outtype q8_0
  ```
3. Then there will be a converted file with .gguf extension.

## Inference in chat mode
You can find this in the [llama.cpp](https://github.com/ggerganov/llama.cpp) here too.

We used alpaca prompt for the test. [our inference code](https://github.com/Aprilistic/llama.cpp/blob/AIDoc_baseline/inference_test/chat.sh)



