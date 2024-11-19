# Finetuning mistral 7b:

import random
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

HF_TOKEN = "XXXXXXXXXXXXXXXXXXXXXX"
from huggingface_hub import login
login(token = HF_TOKEN)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a small query give me what the bigger questions the smaller query could have come from.


### Query:
{data_point["query"]}


### Description:
{data_point["description"]}
"""
    return tokenize(full_prompt)


# Using clueweb09/catb/trec-web-2009 (50 queries with descriptions)
# Using disks45/nocr/trec-robust-2004 (250 queries with descriptions)
# Using gov/trec-web-2002 (50 queries with descriptions)
import ir_datasets
dataset1 = ir_datasets.load("clueweb09/catb/trec-web-2009")
dataset2 = ir_datasets.load("disks45/nocr/trec-robust-2004")
dataset3 = ir_datasets.load("gov/trec-web-2002")

all_data = []

for query in dataset1.queries_iter():
    all_data.append({'query':query[1], 'description':query[2]})

for query in dataset2.queries_iter():
    all_data.append({'query':query[1], 'description':query[2]})

for query in dataset3.queries_iter():
    all_data.append({'query':query[1], 'description':query[2]})

random.shuffle(all_data)

i=0
train_data, eval_data, test_data = [], [], []
for query in all_data:
    if i<280:
      #train_data.append({'query':query[1], 'description':query[2]})
      train_data.append({'query':query['query'], 'description':query['description']})
    elif i<315:
      #eval_data.append({'query':query[1], 'description':query[2]})
      eval_data.append({'query':query['query'], 'description':query['description']})
    else:
      #test_data.append({'query':query[1], 'description':query[2]})
      test_data.append({'query':query['query'], 'description':query['description']})
    i+=1

print("TEST DATA SET:")
print(test_data)

from datasets import Dataset
data_dict = {key: [dic[key] for dic in train_data] for key in train_data[0]}
# Create a Dataset object
train_dataset = Dataset.from_dict(data_dict)

data_dict = {key: [dic[key] for dic in eval_data] for key in eval_data[0]}
# Create a Dataset object
eval_dataset = Dataset.from_dict(data_dict)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

#eval_prompt = """Given a small query give me what the bigger questions the smaller query could have come from.


#### Query:
#getting organized


#### Description:
#"""

#### Description:
#"""

#model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#model.eval()
#with torch.no_grad():
#    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))

# Adding LoRA adapaters to the linear layers of the model

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)
# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

#print(model)

# !pip install wandb

# Start training the fine tuned model:

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

import transformers
from datetime import datetime


project = "finetune-all-combined"
#project =  "finetune-gov-trec-web-2002"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


tokenizer.pad_token = tokenizer.eos_token


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        # max_steps=1000,
        max_steps=100,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        # logging_steps=50,
        logging_steps=20,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        # save_steps=50,                # Save checkpoints every 50 steps
        save_steps=20,
        evaluation_strategy="steps", # Evaluate the model every logging step
        # eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        eval_steps=20,
        do_eval=True,                # Perform evaluation at the end of training
        # report_to="wandb",           # Comment this out if you don't want to use weights & baises
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# 24 steps require 37 mins for 50 datapoints in clueweb09

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    token=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Trying the trained model:

#from peft import PeftModel
#ft_model = PeftModel.from_pretrained(base_model, "mistral-finetune-trec-robust-2004/checkpoint-10")
