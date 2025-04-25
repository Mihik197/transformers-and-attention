import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace  # rule for splitting text based on spaces, tabs, etc

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:  # ds: dataset
        yield item['translation'][lang]  # yields sentences one by one, memory efficient as dataset could be huge

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))  # becomes "tokenizer_en.json" if lang is 'en'
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()  # split text on whitespace initially
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)  # only include words that appear at least twice
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)  # training tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split="train")
    
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config["lang_target"])

    # from train split, we keep 90% for training and 10% for val
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config["lang_src"], config["lang_target"], config["seq_length"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_target, config["lang_src"], config["lang_target"], config["seq_length"])

    max_len_src = 0
    max_len_target = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        target_ids = tokenizer_target.encode(item["translation"][config["lang_target"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_target}')

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target