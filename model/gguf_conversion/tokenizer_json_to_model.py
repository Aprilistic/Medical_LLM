from transformers import PreTrainedTokenizerFast

# load the tokenizer from the json file
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# save the tokenizer to a model file
fast_tokenizer.save_pretrained("tokenizer_directory")