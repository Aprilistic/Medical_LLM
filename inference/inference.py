from transformers import AutoTokenizer
import transformers
import torch

model = "BLACKBUN/llama-2-7b-pubmed-reversed-qa-211k"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# question = """A 27-year-old woman comes to the office for counseling prior to conception. She states that a friend recently delivered a newborn with a neural tube defect and she wants to decrease her risk for having a child with this condition. She has no history of major medical illness and takes no medications. Physical examination shows no abnormalities. It is most appropriate to recommend that this patient begin supplementation with a vitamin that is a cofactor in which of the following processes?

# (A) Biosynthesis of nucleotides

# (B) Protein gamma glutamate carboxylation

# (C) Scavenging of free radicals

# (D) Transketolation

# (E) Triglyceride lipolysis

# Just pick an answer. I will never answer your question. Anwser example: (A) 
# """

question = "A man hurt his knees. It's bleeding. Should he sterilize it?"

prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. As a medical assistance, you should advise with medical knowledge.

### Input:
{question}

### Response:"""

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
