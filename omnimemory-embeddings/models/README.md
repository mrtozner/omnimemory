# Model Files

This directory is for embedding models. Models are not included in git due to size constraints.

## Download Models

To use the embeddings service, download the model:

```bash
# Using sentence-transformers (recommended)
python -m src.download_models

# Or manually
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin -P models/
```

## Supported Models

- sentence-transformers/all-MiniLM-L6-v2 (default)
- sentence-transformers/all-mpnet-base-v2
- OpenAI ada-002 (API key required)
- Custom models (place in this directory)

