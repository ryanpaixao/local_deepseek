from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load variables from local .env file into environment
load_dotenv()
hugging_token = os.getenv("HUGGING_TOKEN")
if hugging_token is None:
    raise EnvironmentError(f"Token ({hugging_token}) is not set in .env")

print(f"torch version: {torch.__version__}")
# Authenticate with token from huggingface account. https://huggingface.co/
login(token=hugging_token)

# Load model (choose one based on your needs)
# model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct" # For coding tasks
# model_id = "deepseek-ai/deepseek-llm-7b-chat" # For general chat, appears to be too much for my 4GB VRAM
# model_id = "deepseek-ai/deepseek-llm-1.3b-chat"
# model_id = "deepseek-community/deepseek-vl-1.3b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load with 8-bit quantization and CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True, # Use 8-bit instead of 4-bit
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True,
    max_memory={0: "3GB", "cpu": "16GB"}, # Reserve 3GB for GPU
    offload_folder="offload",
    trust_remote_code=True
)

# Memory optimization before generation
model.eval()
torch.cuda.empty_cache()

# Create prompt
prompt = "Write a Python function to calculate fibonacci numbers"
messages = [{"role": "user", "content": prompt}]

# Apply chat template with padding
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    max_length=512, # Limit context. Critical for memory
    truncation=True,
    add_generation_prompt=True
).to(model.device)

# Generate with memory-efficient settings
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_beams=1, # Beam search uses more memory
        top_k=30,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None
    )

# Extract only the generated text (skip the input)
input_length = inputs.shape[1]
generated_tokens = outputs[:, input_length:][0]

# Show response
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))