import torch
from transformers import AutoTokenizer
from train_prism import build_dataloader, CONFIG
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dl = build_dataloader(tokenizer, CONFIG)
print("Dataloader built")
for i, batch in enumerate(dl):
    print(batch["input_ids"].shape)
    if i == 2: break
