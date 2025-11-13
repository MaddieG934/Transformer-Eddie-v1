import torch
import torch.nn as nn

import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import datasets
from datasets import load_dataset

from train import get_or_build_tokenizer

from model import Transformer, build_transformer
from config import get_weights_file_path, get_config

def interface():
    src_text = input("Enter a phrase in English: ")
    print("Text received")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model
config = get_config()
ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], config['d_model']).to(device)

# Load checkpoint
checkpoint_path = get_weights_file_path(config, "19")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Restore state
model.load_state_dict(checkpoint["model_state_dict"])

start_epoch = checkpoint["epoch"] + 1       # resume at next epoch
global_step = checkpoint["global_step"]     # resume logging / scheduling

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

interface()