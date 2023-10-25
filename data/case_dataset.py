from datasets import load_dataset
import os
import huggingface_hub

huggingface_hub.login(token='hf_key_here')

ds = load_dataset("csv", data_files='path/cases.csv')
ds = ds['train']
ds.train_test_split(test_size=0.1)

ds.push_to_hub("imaginary_patient_cases")