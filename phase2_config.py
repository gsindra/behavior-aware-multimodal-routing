"""
================================================================================
PHASE 2 — CONFIGURATION
================================================================================
Single source of truth for all paths, persona definitions, hyperparameters,
and confirmed results. Import this in any Phase 2 script.

Author: Indramuthu Sundaram — Phase 2, North Carolina A&T State University
================================================================================
"""

from pathlib import Path
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\gsind\North Carolina A&T State University"
               r"\Marwan Bikdash - Indra\Indra - Research Folder")
GRAPH_F = BASE / "Data"    / "Phase 3" / "G_phase3_rebuilt.pickle"   # 110,324 nodes
GRAPH_S = BASE / "Data"    / "Phase 3" / "G_phase3_small.pickle"     # 4,752 nodes
ORACLE  = BASE / "Results" / "Phase 2 Rerun Clean" / "Phase2_Oracle" / "phase2_oracle_500dest.pickle"
RESULTS = BASE / "Results" / "Phase 2 Rerun Clean"
FIGURES = RESULTS / "figures"
MYCODE  = BASE / "MyCode"  / "Phase 2"

# ── Persona definitions ────────────────────────────────────────────────────────
PERSONA_NAMES = {
    0: "Safety-Conscious Commuter",
    1: "Scenic Walker",
    2: "Efficient Minimalist",
}

# Confirmed 10D behavioral vectors from Phase 1 agglomerative clustering (k=3)
# Trait order: Urgency, Crowd_Love, Crowd_Aversion, Walk_Love, Walk_Aversion,
#              Safety_Sensitivity, Scenic_Love, Scenic_Aversion,
#              Reliability_Expectation, Cost_Sensitivity
BEHAVIOR_VECTORS = {
    0: np.array([0.8724, 0.0514, 0.8565, 0.0517, 0.7305,
                 0.6984, 0.0187, 0.9111, 0.9306, 0.7011], dtype=np.float32),
    1: np.array([0.2577, 0.6485, 0.1740, 0.6020, 0.1628,
                 0.2633, 0.5536, 0.2107, 0.5857, 0.1449], dtype=np.float32),
    2: np.array([0.7878, 0.1303, 0.7118, 0.2033, 0.4590,
                 0.6557, 0.1498, 0.6867, 0.8446, 0.5483], dtype=np.float32),
}

TRAIT_NAMES = [
    'Urgency', 'Crowd_Love', 'Crowd_Aversion', 'Walk_Love', 'Walk_Aversion',
    'Safety_Sensitivity', 'Scenic_Love', 'Scenic_Aversion',
    'Reliability_Expectation', 'Cost_Sensitivity',
]

# ── Clustering metadata (Phase 1) ─────────────────────────────────────────────
CLUSTERING = {
    'algorithm':        'AgglomerativeClustering',  # NOT k-means
    'linkage':          'ward',
    'k':                3,
    'silhouette':       0.59,
    'davies_bouldin':   0.60,
    'embedding_model':  'text-embedding-3-large',
    'embedding_dim':    3072,
    'pca_components':   100,
    'umap_components':  50,
    'trait_weight':     1.5,   # traits upweighted in fusion
    'n_samples_raw':    1247,
    # n_samples_filtered: check console output from Unipolar3kfinalB.py
}

# ── PPO hyperparameters (actual training values) ───────────────────────────────
# NOTE: Paper Table 2 had errors — this is the ground truth from phase2_train.py
PPO_HYPERPARAMS = dict(
    learning_rate = 1e-4,    # paper said 3e-4  ← WRONG
    n_steps       = 4096,    # paper omitted    ← ADD TO TABLE
    batch_size    = 256,     # paper said 64    ← WRONG
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.20,
    ent_coef      = 0.005,   # paper said 0.01  ← WRONG
    vf_coef       = 0.50,
    max_grad_norm = 0.50,
    verbose       = 1,
)

# ── Variant configuration ─────────────────────────────────────────────────────
VARIANT_CONFIG = {
    'v1s': {
        'name':    'V1s — Pure PPO (small graph)',
        'graph':   'small',    # 4,752 nodes
        'top_k':   6,
        'bc':      False,
        'oracle':  False,
    },
    'v2oracle': {
        'name':    'V2 — Oracle BC-PPO (full graph)',
        'graph':   'full',     # 110,324 nodes
        'top_k':   10,
        'bc':      True,
        'oracle':  True,
    },
    'v3': {
        'name':    'V3 — Route Selection (full graph)',
        'graph':   'full',
        'top_k':   6,
        'bc':      False,
        'oracle':  False,
    },
    'bwd': {
        'name':    'BWD — Behavior-Weighted Dijkstra',
        'graph':   'full',
        'top_k':   0,
        'bc':      False,
        'oracle':  False,
    },
}

TRAINING_STEPS = 800_000

# ── Confirmed results (verified against eval CSVs, N=200) ────────────────────
# These are the ground-truth numbers. Do not change without re-running eval.
CONFIRMED_RESULTS = {
    'v1s': {
        0: {'strict': 0.5,   'ci': (0.1, 2.8)},
        1: {'strict': 0.0,   'ci': (0.0, 1.9)},
        2: {'strict': 0.0,   'ci': (0.0, 1.9)},
        'js': {(0,1): 0.0027, (0,2): 0.0033, (1,2): 0.0001},
    },
    'v2oracle': {
        0: {'strict': 4.5,   'ci': (2.4, 8.3)},
        1: {'strict': 2.0,   'ci': (0.8, 5.0)},
        2: {'strict': 100.0, 'ci': (98.1, 100.0)},
        'js': {(0,1): 0.6086, (0,2): 0.1310, (1,2): 0.4315},
    },
    'v3': {
        0: {'strict': 100.0, 'ci': (98.1, 100.0)},
        1: {'strict': 100.0, 'ci': (98.1, 100.0)},
        2: {'strict': 100.0, 'ci': (98.1, 100.0)},
        'js': {(0,1): 0.1114, (0,2): 0.1290, (1,2): 0.0179},
    },
    'bwd': {
        0: {'strict': 100.0, 'ci': (98.1, 100.0)},
        1: {'strict': 100.0, 'ci': (98.1, 100.0)},
        2: {'strict': 100.0, 'ci': (98.1, 100.0)},
        'js': {(0,1): 0.0746, (0,2): 0.0560, (1,2): 0.1961},
    },
}

# ── Mode attributes (environment constants) ────────────────────────────────────
MODE_ATTRS = {
    'walk':           {'safety':0.6,'scenic':0.9,'reliability':0.7,'crowd':0.3,'cost':0.0},
    'bike':           {'safety':0.5,'scenic':0.8,'reliability':0.6,'crowd':0.2,'cost':0.0},
    'bus':            {'safety':0.7,'scenic':0.6,'reliability':0.5,'crowd':0.8,'cost':0.3},
    'subway':         {'safety':0.8,'scenic':0.3,'reliability':0.9,'crowd':0.9,'cost':0.3},
    'car':            {'safety':0.9,'scenic':0.5,'reliability':0.85,'crowd':0.1,'cost':0.8},
    'ride-hail':      {'safety':0.9,'scenic':0.4,'reliability':0.8,'crowd':0.1,'cost':1.0},
    'transit_access': {'safety':0.7,'scenic':0.6,'reliability':0.7,'crowd':0.4,'cost':0.0},
}


def make_personas_df():
    """Build personas DataFrame from confirmed behavior vectors."""
    import pandas as pd
    return pd.DataFrame([
        {'cluster': pid, **dict(zip(TRAIT_NAMES, vec.tolist()))}
        for pid, vec in BEHAVIOR_VECTORS.items()
    ])


if __name__ == "__main__":
    print("Phase 2 Configuration")
    print(f"  Graph (full):  {GRAPH_F}")
    print(f"  Graph (small): {GRAPH_S}")
    print(f"  Results:       {RESULTS}")
    print(f"\nPersonas:")
    for pid, name in PERSONA_NAMES.items():
        print(f"  P{pid}: {name}")
        print(f"       b = {BEHAVIOR_VECTORS[pid]}")
    print(f"\nClustering: {CLUSTERING['algorithm']} k={CLUSTERING['k']} "
          f"SC={CLUSTERING['silhouette']} DBI={CLUSTERING['davies_bouldin']}")
    print(f"\nPPO: lr={PPO_HYPERPARAMS['learning_rate']} "
          f"batch={PPO_HYPERPARAMS['batch_size']} "
          f"ent={PPO_HYPERPARAMS['ent_coef']}")
