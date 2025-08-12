from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configuration for 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model (choose one based on your needs)
# model_id = "deepseek-ai/deepseek-llm-7b-chat" # For general chat, appears to be too much for my 4GB VRAM
# model_id = "deepseek-ai/deepseek-coder-6.7b-instruct" # For coding tasks
model_id = "deepseek-ai/deepseek-llm-1.3b-chat"

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True # Reduces memory footprint
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    # llm_int8_enable_fp32_cpu_offload=True,
    # max_memory={0: "4GB", "cpu": "16GB"},
    trust_remote_code=True
)

# Create prompt
prompt = "Explain quantum computing in simple terms"
messages = [{"role": "user", "content": prompt}]

# Generate response
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.15
)

# Show response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))