from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model

import os
from dotenv import load_dotenv

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

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True # Reduces memory footprint
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 1. Create empty model to inspect architecture
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )

# 2. Create aggressive device map - only first 4 layers on GPU
gpu_id = 0
def add_model_layers() -> dict:
    return {f"model.layers.{i}": "cpu" for i in range(4, 32)} # Add model layers for cpu up to "model.layers.31"

# print(f"add func: {add_model_layers(31)}")
device_map = {
    "model.embed_tokens": gpu_id,
    "model.layers.0": gpu_id,
    "model.layers.1": gpu_id,
    "model.layers.2": gpu_id,
    "model.layers.3": gpu_id
}
device_map.update(add_model_layers())
device_map.update({"model.norm": "cpu", "lm_head": "cpu"})

# 3. Load model with manual offloading
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # llm_int8_enable_fp32_cpu_offload=True,
    quantization_config=quant_config,
    device_map=device_map,
    offload_state_dict=True, # Critical for 4GB VRAM
    offload_folder="offload",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 4. Memory optimization before generation
model.enable_input_require_grads()
torch.cuda.empty_cache()

# Create prompt
prompt = "Write a Python function to calculate fibonacci numbers"
messages = [{"role": "user", "content": prompt}]

# Apply chat template with padding
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512, # Limit context
    add_generation_prompt=True
    # return_attention_mask=True
).to(model.device)

# Generate with strict memory constraints
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_k=30,
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.2,
        num_beams=1, # Beam search uses more memory
        pad_token_id=tokenizer.eos_token_id,
    )

# Extract only the generated text (skip the input)
input_length = inputs.shape[1]
generated_tokens = outputs[:, input_length:][0]

# Show response
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))