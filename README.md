# KGSynX

KGSynX is a Knowledge Graph and SHAP-guided synthetic tabular data generation framework. This repository contains core modules to construct knowledge graphs, train embeddings, generate LLM-based synthetic data, and refine generations iteratively based on SHAP explanations.

## Folder Structure

- `load_data.py`: Load and preprocess real datasets (e.g., UCI Heart Disease).
- `build_kg.py`: Construct patient-level knowledge graph from labeled tabular data.
- `embedding.py`: Learn node embeddings using Node2Vec.
- `train_and_shap.py`: Train classifiers and compute SHAP explanations.
- `refinement_loop.py`: Iteratively generate synthetic data with feedback loop.
- `utils.py`: Common utility functions (e.g., validation, similarity computation).
- `main.py`: End-to-end pipeline to run KGSynX on UCI Heart Disease dataset.

## Setup

```bash
pip install -r requirements.txt
```

Required packages include:
- `pandas`, `scikit-learn`, `networkx`
- `gensim`, `matplotlib`, `shap`, `openai`

## Usage

You can run the full pipeline using:

```bash
python main.py
```

This will:
1. Load and preprocess the dataset.
2. Build the knowledge graph.
3. Compute node embeddings.
4. Generate synthetic samples using GPT-based model.
5. Refine generations based on SHAP feedback until convergence.

## Notes

- This version uses OpenAI GPT API for text generation. You must configure your API key via environment variables or securely within your workflow.
- Sample outputs and synthetic datasets are saved to `./outputs`.

## Citation

If you use this code in your research, please cite our paper:

> [Paper Title]  
> [Authors]  
> [ISWC 2025, under review]

