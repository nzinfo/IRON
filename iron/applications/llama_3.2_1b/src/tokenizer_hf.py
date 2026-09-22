# SPDX-License-Identifier: Apache-2.0
# MiniCPM5 (GPT-2 BPE) tokenizer shim exposing the app's tiktoken-style
# Tokenizer interface: encode(text, bos, eos) / decode(ids) / .special

from tokenizers import Tokenizer

BOS_ID = 0      # "<s>"
EOS_ID = 1      # "</s>" (config eos list: [1, 130073])


class HFTokenizer:
    def __init__(self, model_path):
        self.model = Tokenizer.from_file(model_path)
        self.special = {
            "<|begin_of_text|>": BOS_ID,
            "<|end_of_text|>": EOS_ID,
        }

    def encode(self, text, bos=False, eos=False):
        ids = self.model.encode(text).ids
        if bos:
            ids = [BOS_ID] + ids
        if eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids):
        return self.model.decode(ids, skip_special_tokens=True)
