# Transformer-based Character-Level Language Model

This project implements a character-level Transformer language model in PyTorch. The model is trained from scratch on a text file and can generate text in the style of the training data.

## 🚀 Features

- Transformer architecture with multi-head self-attention
- Character-level tokenization and embeddings
- Causal masking for autoregressive generation
- Position embeddings
- Configurable number of layers, heads, embedding dimensions
- Training loop with evaluation on train/val split
- Text generation from trained model

## 🧠 Model Architecture

- Embedding layer for tokens and positions
- `n_layer` Transformer blocks, each with:
  - Multi-head self-attention
  - Feedforward layers
  - Residual connections + LayerNorm
- Linear layer to vocab size to get logits

## 🧪 Training

- Loss: CrossEntropyLoss (character prediction)
- Optimizer: AdamW
- Batch size: 64
- Block size (context length): 256
- Learning rate: 3e-4
- Dropout: 0.2
- Training steps: 5000 (default)

## 🧠 Inspiration & References

This project is based on the Transformer architecture introduced in  
["Attention is All You Need" by Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)  
and is inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — a minimal educational GPT implementation.


## 🧾 Usage

### 1. Prepare data

Place your training text in a file called `input.txt`.

### 2. Run training

```bash
python your_script_name.py
