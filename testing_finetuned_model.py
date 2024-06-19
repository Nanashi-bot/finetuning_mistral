# Trying the trained model:
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    token=True
)

from peft import PeftModel
#ft_model = PeftModel.from_pretrained(base_model, "/b/aditya/mistral-mistral-finetune-clueweb09/checkpoint-100")
ft_model = PeftModel.from_pretrained(base_model, "/b/aditya/mistral-finetune-trec-robust-2004/checkpoint-50")
#ft_model = PeftModel.from_pretrained(base_model, "/b/aditya/mistral-finetune-gov-trec-web-2002/checkpoint-100")

ft_model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=20, pad_token_id=2)[0], skip_special_tokens=True))

import ir_datasets
dataset1 = ir_datasets.load("clueweb09/catb/trec-web-2009")
dataset2 = ir_datasets.load("disks45/nocr/trec-robust-2004")
dataset3 = ir_datasets.load("gov/trec-web-2002")

#data = []
#for query in dataset.queries_iter():
#    data.append({'query':query[1], 'description':query[2]})

test_data = []

for i in range(35):

    eval_prompt = f"""Given a small query give me what the bigger questions the smaller query could have come from.


### Query:
{test_data[i]["query"]}


### Description:
"""
#    if i == 0:
#       print()
#       print("TRAINING DATASET: ")
#    if i == 40:
#        print()
#        print("EVALUATION DATASET: ")
#    if i == 45:
#        print()
#        print("TEST DATASET: ")


    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    print("QUERY: ")
    print(test_data[i]["query"])
    print("-"*70)
    print("DESCRIPTION THAT MISTRAL GENERATES: ")
    with torch.no_grad():
        #print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=30, pad_token_id=2)[0], skip_special_tokens=True).split("Description:")[1])
        print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=35, pad_token_id=2)[0], skip_special_tokens=True).split("Description:")[1].split("###")[0])
    print("-"*70)
    print("DESCRIPTION FROM DATASET: ")
    print(test_data[i]["description"])
    print("-"*70)
