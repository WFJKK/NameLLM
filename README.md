# Character-Level Name Generator

A (modular) character-level Transformer trained on the standard `names.txt`. This project implements modern decoder-only Transformer using PyTorch and supports training with masked causal attention, padding-aware loss, and name sampling via autoregression. Includes modern architecture elements such as pre layernorm, feedforward, dropout, residual connections.

---

##  Project Structure

- `/namegen`: main package files
- `generating_names.py`: script for generating names
- `requirements.txt`: dependencies
- `names.txt`: text file of names used for training and validation



---

##  Features

- Transformer-style decoder-only architecture
-  Causal masking and padding-aware loss
-  Modular, readable PyTorch implementation
-  Autoregressive name generation
-  W&B integration for logging
-  Checkpointing and resume support

---

##  Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/char-transformer-namegen.git
   cd char-transformer-namegen
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Training

To train the model on `names.txt` w/:

 training.py


- Automatically uses GPU (MPS or CUDA) if available
- Logs training loss to [Weights & Biases](https://wandb.ai/)
- Automatically checkpoints to `checkpoint.pth` every 10 epochs
- Resumes from checkpoint if one exists

Modify `config.py` to adjust hyperparameters such as :
- Model dimensions
- Number of layers/heads
- Learning rate, batch size, etc.

---

##  Generate Names

After training you can sample new names w/:

 generating_names.py

This calls the name_generate function defined in `name_generation.py`, which uses the trained model to generate names autoregressively one character at a time.

---



## License

This project is licensed under the **MIT License** — feel free to use, modify, or contribute.

---

