from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_length):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_length = seq_length

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2  # 2 for SOS and EOS
        dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1  # 1 for SOS

        if enc_num_padding_tokens < 0 or  dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # SOS, source_text, EOS, PADDING
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # SOS, target_text, PADDING
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # shifted by one i.e. target_text, EOS, PADDING
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_length)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, 1, seq_length) & (1, seq_length, seq_length)
            "label": label,
            "src_text": src_text,
            "target_text": target_text,
        }

def causal_mask(size):
    # resulting mask is True on and below the diagonal and False strictly above
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # diagonal=1 shifts boundary up by one from main diagonal, so you only keep the elements strictly above the main diagonal
    return mask == 0