"""
Defines a PyTorch Dataset for character-level name sequences with padding, tokenization, and positional encoding.
"""

import torch
from torch.utils.data import Dataset


class NameDataset(Dataset):
    """
    A PyTorch Dataset for character-level name modeling.

    Each name is wrapped with start ('.') and end ('.') tokens.
    The dataset provides access to raw names and supports a static
    collate function that performs padding and token/position encoding
    for a batch of names.

    Attributes:
        names (List[str]): List of names, each prepended and appended with '.'.
        stoi (Dict[str, int]): A dictionary mapping characters to token indices.
        max_len (int): Maximum name length in the dataset (including '.' tokens).
        padding_idx (int): Index used for padding tokens.

    Methods:
        __len__(): Returns the number of names in the dataset.
        __getitem__(idx): Returns the raw name at the given index.
        collate_fn(batch_names, stoi, padding_idx): Static method for padding
            a batch of names and converting them to token and position tensors.
    """

    def __init__(self, names, stoi, padding_idx):
        self.names = ["." + name + "." for name in names]
        self.stoi = stoi
        self.max_len = max(len(name) for name in self.names)
        self.padding_idx = padding_idx

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx]

    @staticmethod
    def collate_fn(batch_names, stoi, padding_idx):
        max_len_batch = max(len(name) for name in batch_names)

        token_batches = []
        pos_batches = []
        for name in batch_names:
            tokens = [stoi[c] for c in name]
            pad_len = max_len_batch - len(tokens)
            tokens = tokens + [padding_idx] * pad_len
            pos_tokens = [i + 1 for i in range(len(name))] + [0] * pad_len

            token_batches.append(torch.tensor(tokens, dtype=torch.long))
            pos_batches.append(torch.tensor(pos_tokens, dtype=torch.long))

        return torch.stack(token_batches), torch.stack(pos_batches)
