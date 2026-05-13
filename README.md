> The scientific idea was introduced by Jędrzej Kubica, Shashank Katiyar, and Ben Busby at the CMU / DNAnexus Hackathon on October 19–21, 2023. The technology was created by agentic AI provided by Claude Sonnet 4.6. The original project is available under the MIT license at https://github.com/collaborativebioinformatics/virussusceptibility

# Virus Susceptibility

A Python pipeline to build a vector database from the [CORD-19 dataset](https://github.com/allenai/cord19)
and query it for relevant scientific articles using semantic similarity.

Originally developed at the CMU / DNAnexus Hackathon, October 2023.

## Installation

Python 3.9+ is required.

```bash
# CPU-only
pip install numpy>=1.23 pandas>=1.5 transformers>=4.30 scikit-learn>=1.3 \
    torch>=2.0 --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8)
pip install numpy>=1.23 pandas>=1.5 transformers>=4.30 scikit-learn>=1.3 \
    torch>=2.0 --index-url https://download.pytorch.org/whl/cu118

# Optional: FAISS backend
pip install faiss-cpu>=1.7      # CPU
# pip install faiss-gpu>=1.7    # GPU
```

## Usage

### 1. Download data

```bash
python scripts/download_data.py --output-dir data/ --release 2022-06-02
```

### 2. Build the vector database

```bash
python scripts/build_database.py \
    --data-dir   data/ \
    --output-dir db/ \
    --release    2022-06-02
```

For older releases without pre-computed embeddings (e.g. 2020-03-13), the script
embeds titles and abstracts automatically using SPECTER. Use `--max-articles N`
to limit the number of articles embedded.

Optional flags: `--model <hf-model-id>`, `--db-backend faiss`.

### 3. Query the database

```bash
python scripts/query_database.py \
    --db-file  db/vector_db.pkl \
    --data-dir data/ \
    --release  2022-06-02 \
    --query    "hypertension and COVID-19"
```

Use `--top-k N` to change the number of results (default: 10).
Use `--model <hf-model-id>` to query with a custom embedding model.

## Data

CORD-19 (COVID-19 Open Research Dataset) — [https://github.com/allenai/cord19](https://github.com/allenai/cord19)
