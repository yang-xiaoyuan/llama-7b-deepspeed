import torch 
from transformers import LlamaForCausalLM, AutoTokenizer

base_model_path = './llama-2-7b-hf'
merged_model_path = './llama-2-7b-merged'

tokenizer = tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

test_prompt = """
Description: Q. Every time I eat spicy food, I poop blood. Why?
Patient: Hi doctor, I am a 26 year old male. I am 5 feet and 9 inches tall and weigh 255 pounds. When I eat spicy food, I poop blood. Sometimes when I have constipation as well, I poop a little bit of blood. I am really scared that I have colon cancer. I do have diarrhea often. I do not have a family history of colon cancer. I got blood tests done last night. Please find my reports attached.
Doctor: 
"""

test_prompt = """
Description: What causes respiratory problem in a 36-year-old?
Patient: Hello doctor, My mother is 36 years old, and she is suffering from the respiratory problem from last two days, her hemoglobin count is 9.6, can you please tell what causes this problem?
Doctor: 
"""

model_input = tokenizer(test_prompt, return_tensors='pt').to('cuda')


model = LlamaForCausalLM.from_pretrained(base_model_path, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
model.eval()
with torch.no_grad():
    result = model.generate(**model_input, max_new_tokens=150)[0]
    print("---------------Result Before Finetuned---------------")
    print(tokenizer.decode(result, skip_special_tokens=True))


model = LlamaForCausalLM.from_pretrained(merged_model_path, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
model.eval()
with torch.no_grad():
    result = model.generate(**model_input, max_new_tokens=150)[0]
    print("---------------Result After Finetuned---------------")
    print(tokenizer.decode(result, skip_special_tokens=True))