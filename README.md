# RAW: Refusal Aversion in the Wild

Pipeline code for the paper *RAW: Refusal Aversion in the Wild, A Causal
Measurement Method for Deployed LLMs* (EMNLP 2026 Industry Track). RAW uses
LLM sampling stochasticity as a natural experiment: at near-identical prompts
in an existing conversation log, the same model sometimes refuses and
sometimes does not, and that variation identifies the causal effect of a
refusal on user re-engagement from logs alone.

This repository contains the full pipeline: extraction, refusal labelling,
prompt embedding, matching, cross-fit AIPW estimation with cluster-robust
inference, Rosenbaum sensitivity, and paper figures. The derived data
artifacts (refusal labels, moderation scores, prompt embeddings, and
conversation-hash join keys; no raw text) are released separately:

- HuggingFace dataset: https://huggingface.co/datasets/pkireyev1/raw-refusal-aversion-in-the-wild
- Permanent identifier: https://doi.org/10.5281/zenodo.22073974

## Data requirements

The source corpora are gated and are not redistributed here. Obtain them from
their maintainers and point the paths at the top of the setup scripts (and
`llm_dynamics/config.py`) at your local copies:

- `lmsys/lmsys-chat-1m` (Hugging Face, gated; custom license)
- `allenai/WildChat-4.8M-Full` (Hugging Face, gated; ODC-BY 1.0)

Labelling requires an OpenAI API key (`OPENAI_API_KEY`) for the
`gpt-5.4-mini` judge via the Batch API; the optional open-weights fallback
judge (`allenai/wildguard`) needs `torch` + `transformers` (see
`requirements.txt`). Embedding uses `mxbai-embed-large` served locally via
Ollama (or any drop-in embedding endpoint; see `embed_direct.py`).

## Pipeline

Run stages in order per cell (a model on one conversation pool):

| Stage | Script |
|---|---|
| 1. Extraction (LMSYS cells) | `setup_lmsys.py`, `setup_claude1.py` |
| 1. Extraction (WildChat cells) | `setup_wildchat.py`, `setup_gpt4o.py`, `setup_wildchat_risky.py`, `merge_wildchat_aug.py` |
| 2. Refusal labelling | `label_with_openai_batch.py` (judge), `label_with_wildguard.py` (fallback) |
| 3. Prompt embedding | `embed_direct.py` |
| 4. Matching estimators | `matching_with_gpt54.py` |
| 5. Cross-fit AIPW | `aipw_with_gpt54.py` |
| 6. Rosenbaum sensitivity | `sensitivity_with_gpt54.py` |
| 7. Figures | `figures/*.py` |

### Inference regimes

`aipw_with_gpt54.py` computes three standard-error regimes: unclustered
(conversation-level iid), `--cluster ip` (hashed-IP user clusters, WildChat
cells; requires `--meta-parquet` with per-conversation hashed IPs), and
`--cluster promptcluster` (near-duplicate prompt groups). The paper's default
regime is user clusters wherever the source data identify users and prompt
groups otherwise. After running the cluster variants, assemble the
per-cell default intervals for the figure scripts with:

```
python collect_default_inference.py --cells-dir output/comparison/aipw_clustered/cells
```

The same script also supports covariate robustness variants
(`--add-time`, `--add-temp`, `--restrict-modal-temp`, `--length-stratum`).

## Reproducibility

Every K-fold split, gradient-boosted-tree fit, and matching step is seeded;
re-running the pipeline reproduces the published numbers up to numerical
noise from BLAS thread scheduling. `requirements.txt` pins the versions used
for the released results. Matching and estimation run on CPU in minutes per
cell.

## Citation

```bibtex
@inproceedings{kireyev2026raw,
  title = {{RAW}: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed {LLMs}},
  author = {Kireyev, Pavel},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing: Industry Track},
  year = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Derived data follow
each source corpus's license; WildChat-derived artifacts are ODC-BY 1.0 with
attribution to WildChat, and no LMSYS-Chat-1M content is redistributed.
