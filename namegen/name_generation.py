"""
Provides a function to generate names token-by-token from a trained Transformer model using temperature sampling.
"""

import torch
import torch.nn.functional as F


def name_generate(model, stoi, itos, max_length, padding_idx, device, temperature=1.0):
    """
    Generate a name sequence from a trained Transformer model, starting from a start token.

    Args:
        model (nn.Module): Trained Transformer model.
        stoi (dict): Dictionary mapping characters/tokens to indices.
        itos (dict): Dictionary mapping indices to characters/tokens.
        max_length (int): Maximum length of the generated sequence.
        padding_idx (int): Index of padding token.

    Returns:
        str: Generated name as a string.
    """
    model.eval()
    generated = [stoi["."]]

    for _ in range(max_length):
        tokens = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        positions = torch.tensor(
            [list(range(1, len(generated) + 1))], dtype=torch.long
        ).to(device)

        pad_len = max_length - tokens.shape[1]
        if pad_len > 0:
            tokens = torch.cat(
                [tokens, torch.full((1, pad_len), padding_idx, device=device)], dim=1
            )
            positions = torch.cat(
                [positions, torch.zeros((1, pad_len), dtype=torch.long, device=device)],
                dim=1,
            )

        with torch.no_grad():
            logits = model(tokens, positions)

        last_logits = logits[0, len(generated) - 1]
        probs = F.softmax(last_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

        if next_token == stoi["."]:
            break

    name = "".join(itos[c] for c in generated)
    return name
