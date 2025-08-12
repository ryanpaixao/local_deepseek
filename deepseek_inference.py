from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login

import os
from dotenv import load_dotenv

# Load variables from local .env file into environment
load_dotenv()
hugging_token = os.getenv("HUGGING_TOKEN")
if hugging_token is None:
    raise EnvironmentError(f"Token ({hugging_token}) is not set in .env")

# Authenticate with token from huggingface account. https://huggingface.co/
login(token=hugging_token)

# Load model (choose one based on your needs)
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
# model_id = "deepseek-ai/deepseek-coder-6.7b-instruct" # For coding tasks
# model_id = "deepseek-ai/deepseek-llm-7b-chat" # For general chat, appears to be too much for my 4GB VRAM
# model_id = "deepseek-ai/deepseek-llm-1.3b-chat"
# model_id = "deepseek-community/deepseek-vl-1.3b-chat"

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True # Reduces memory footprint
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    # llm_int8_enable_fp32_cpu_offload=True,
    # max_memory={0: "4GB", "cpu": "16GB"},
    trust_remote_code=True
)

# Create prompt
prompt = "Write a Python function to calculate fibonacci numbers"
messages = [{"role": "user", "content": prompt}]

# Generate response
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    return_attention_mask=True
).to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=400,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
    # top_k=40,
    # repetition_penalty=1.15
)

# Extract only the generated text (skip the input)
input_length = inputs.shape[1]
generated_tokens = outputs[:, input_length:][0]

# Show response
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))