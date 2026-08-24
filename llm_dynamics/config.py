"""Global configuration: paths and structural constants.

Paths are resolved relative to the project root (the folder containing this
package). The layout assumes cached intermediates (message-level dataframe,
embeddings, refusal labels) sit alongside the notebook in the project root.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent

# Raw LMSYS-Chat-1M parquet directory (only needed for from-scratch data build).
RAW_LMSYS_DIR: Path = Path("data/lmsys")  # local LMSYS-Chat-1M parquet directory

# Cached intermediates (required).
MESSAGE_DF_PKL: Path = PROJECT_ROOT / "gpt35_message_df_with_users.pkl"
REFUSAL_LABELS_CSV: Path = PROJECT_ROOT / "refusal_data_dynamic_with_labels.csv"
FIRST_USER_EMBEDDINGS_NPZ: Path = PROJECT_ROOT / "first_user_embeddings.npz"
HIGH_CONCERN_EMBEDDINGS_NPY: Path = PROJECT_ROOT / "high_concern_embeddings.npy"
HIGH_CONCERN_METADATA_CSV: Path = PROJECT_ROOT / "high_concern_metadata.csv"

# Where generated figures/tables/intermediates go.
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
EM_RESULTS_DIR: Path = OUTPUT_DIR / "em_results"
BOOTSTRAP_DIR: Path = OUTPUT_DIR / "bootstrap"
CF_DIR: Path = OUTPUT_DIR / "counterfactuals"

for _p in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, EM_RESULTS_DIR, BOOTSTRAP_DIR, CF_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Structural constants (paper defaults)
# ----------------------------------------------------------------------------
# Focal LLM: the paper restricts attention to llama-2-13b-chat responses.
FOCAL_MODEL: str = "llama-2-13b-chat"
# Filter to English conversations only.
LANGUAGE: str = "English"
# Moderation score threshold separating safe (low) from risky (high) segments.
RISK_THRESHOLD: float = 0.01
# Discount factor (fixed following paper; not estimated).
DELTA: float = 0.995
# Caliper for score-only matching.
SCORE_MATCH_CALIPER: float = 0.1
# Max score difference allowed in the second stage of embedding-matched pairs.
EMBEDDING_SCORE_CAP: float = 0.01
# Number of bootstrap replications used for standard errors (paper: 100).
N_BOOTSTRAP: int = 100
# Euler-Mascheroni constant (Type-I Extreme Value normalisation).
EULER_GAMMA: float = 0.5772156649015329
