"""Generate 5 example names using trained decoder-only transformer model."""

import torch
from namegen.config import *
from namegen.data import MAX_SEQ_LEN, PADDING_IDX, itos, stoi, token_dim
from namegen.model import Transformermodel
from namegen.name_generation import name_generate

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = Transformermodel(
    token_dim=token_dim,
    model_dim=model_dim,
    max_seq_length=MAX_SEQ_LEN,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    padding_idx=PADDING_IDX,
)
model = model.to(device)


checkpoint = torch.load("checkpoint.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # set to eval mode

for _ in range(5):
    name = name_generate(
        model=model,
        stoi=stoi,
        itos=itos,
        max_length=MAX_SEQ_LEN,
        padding_idx=PADDING_IDX,
        device=device,
        temperature=1.0,
    )
    print(name)
