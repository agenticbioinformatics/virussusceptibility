import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

SPECTER_MODEL_ID = "allenai/specter"
MAX_TOKEN_LENGTH = 512


def load_model(model_id: str = SPECTER_MODEL_ID):
    """Load any HuggingFace encoder model and return (tokenizer, model).

    Works with any model that outputs last_hidden_state, e.g.:
      - "allenai/specter"  (default, scientific papers)
      - "sentence-transformers/all-MiniLM-L6-v2"
      - "sentence-transformers/all-mpnet-base-v2"
      - a local path to a saved model directory
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def load_specter():
    """Convenience wrapper — loads allenai/specter."""
    return load_model(SPECTER_MODEL_ID)


def embed_paper(paper: dict, tokenizer, model) -> np.ndarray:
    """Embed a single {"title": ..., "abstract": ...} dict; returns shape (hidden_dim,)."""
    text = paper["title"] + tokenizer.sep_token + (paper.get("abstract") or "")
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKEN_LENGTH)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].numpy()[0]


def embed_papers(papers: list, tokenizer, model) -> np.ndarray:
    """Batch-embed a list of {"title": ..., "abstract": ...} dicts; returns shape (N, hidden_dim)."""
    title_abs = [p["title"] + tokenizer.sep_token + (p.get("abstract") or "") for p in papers]
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKEN_LENGTH)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].numpy()


def embed_query(query: str, tokenizer, model) -> np.ndarray:
    """Embed a raw text query string; returns shape (hidden_dim,)."""
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKEN_LENGTH)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].numpy()[0]
