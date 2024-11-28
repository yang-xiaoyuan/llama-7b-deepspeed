import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
import time

# 导入 deepspeed 库和 pytorch 通信后端
import deepspeed
import torch.distributed as dist

# 修改：初始化分布式环境
deepspeed.init_distributed()
world_size = dist.get_world_size()
rank = dist.get_rank()

data_files = {"train": "./data/dialogues.parquet"}
dataset = load_dataset("parquet", data_files=data_files)

def merge_fields(examples):
    examples["text"] = [
        f"Description: {Description} Patient: {Patient} Doctor: {Doctor}" 
        for Description, Patient, Doctor in zip(examples["Description"], examples["Patient"], examples["Doctor"])
    ]
    return examples

dataset = dataset.map(merge_fields, batched=True)

split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  
train_dataset = split_dataset['train']
temp_split = split_dataset['test'].train_test_split(test_size=0.5, seed=42) 
test_dataset = temp_split['test']
eval_dataset = temp_split['train']

# 修改：只有 rank0 进程打印信息
if rank == 0:
    print("Find GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])  # 列出设备
    print("[rank0] Train size:", len(train_dataset))
    print("[rank0] Test size:", len(test_dataset))
    print("[rank0] Eval size:", len(eval_dataset))

output_dir = './output-deepspeed'

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# 修改：训练参数中加入 deepspeed 配置文件
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    optim='adamw_torch',
    learning_rate=10e-4,
    eval_steps=100,
    logging_steps=200,
    eval_strategy='steps',
    group_by_length=False,
    max_steps=200,
    # num_train_epochs=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    deepspeed="./ds_config.json"  # 加入 deepspeed 配置文件
)

model_path = './llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

model.enable_input_require_grads() 
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    dataset_text_field='text'
)

# 修改：只有 rank0 进程打印信息
if rank == 0:
    print("start training")
    start_time = time.time()

trainer.train()

if rank == 0:
    end_time = time.time()
    print("finished training")
    print("time_used:", end_time-start_time, "s")

trainer.model.save_pretrained(output_dir)

