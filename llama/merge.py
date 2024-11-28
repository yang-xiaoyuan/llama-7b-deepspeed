from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

base_model_path = './llama-2-7b-hf'
finetune_model_path = './output'

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(finetune_model_path, device_map='auto', torch_dtype=torch.bfloat16)

model = model.merge_and_unload()

merged_model_path = './llama-2-7b-merged'
model.save_pretrained(merged_model_path)