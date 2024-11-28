import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
import time

# 看一下环境中的 GPU 情况
print("Find GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]) 

# 加载数据集
data_files = {"train": "./data/dialogues.parquet"}
dataset = load_dataset("parquet", data_files=data_files)

# 合并 Description, Patient 和 Doctor 字段，
def merge_fields(examples):
    examples["text"] = [
        f"Description: {Description} Patient: {Patient} Doctor: {Doctor}" 
        for Description, Patient, Doctor in zip(examples["Description"], examples["Patient"], examples["Doctor"])
    ]
    return examples

dataset = dataset.map(merge_fields, batched=True)

# 划分数据集
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  # 划分数据集为训练集和测试集
train_dataset = split_dataset['train']
temp_split = split_dataset['test'].train_test_split(test_size=0.5, seed=42)  # 再从测试集中划分验证集
test_dataset = temp_split['test']
eval_dataset = temp_split['train']

# 打印各数据集的大小
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Eval size:", len(eval_dataset))

output_dir = './output'

# 配置 lora 微调的参数
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# 配置训练的参数
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
    warmup_steps=100
)

# 加载模型及权重
model_path = './llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)


model.enable_input_require_grads()  # 允许权重参与反向传播
model = get_peft_model(model, peft_config)  # 把模型用 PEFT 框架加载，以支持高效微调
model.print_trainable_parameters()
model.config.use_cache = False

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'

# 定义 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    dataset_text_field='text',
)

# 训练，并打印时间
print("start training")
start_time = time.time()
trainer.train()
end_time = time.time()
print("finished training")
print("time_used:", end_time-start_time, "s")

trainer.model.save_pretrained(output_dir)
