"""
================================================================================
PHASE 2 — TRAINER
================================================================================
Trains one routing variant for one persona using PPO.

Usage:
    python phase2_train.py --variant v1s     --persona 0
    python phase2_train.py --variant v2oracle --persona 1 --steps 800000
    python phase2_train.py --variant v3      --persona 2
    python phase2_train.py --all             --steps 800000

Variants:
    v1s      Pure PPO, small graph (4,752 nodes)   [tractability baseline]
    v2oracle BC-PPO + precomputed oracle, full graph [main RL variant]
    v3       Route selection PPO, full graph         [preference learning]

Confirmed hyperparameters (actual training values — verified against results):
    learning_rate = 1e-4   batch_size = 256   ent_coef = 0.005
    n_steps = 4096         gamma = 0.99       clip_range = 0.20

Author: Indramuthu Sundaram — Phase 2, North Carolina A&T State University
================================================================================
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\gsind\North Carolina A&T State University"
               r"\Marwan Bikdash - Indra\Indra - Research Folder")
GRAPH_F = BASE / "Data"    / "Phase 3" / "G_phase3_rebuilt.pickle"   # 110K nodes
GRAPH_S = BASE / "Data"    / "Phase 3" / "G_phase3_small.pickle"     # 4,752 nodes
ORACLE  = BASE / "Results" / "Phase 2 Rerun Clean" / "Phase2_Oracle" / "phase2_oracle_500dest.pickle"
RESULTS = BASE / "Results" / "Phase 2 Rerun Clean"
MYCODE  = BASE / "MyCode"  / "Phase 2"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Persona definitions ────────────────────────────────────────────────────────
PERSONA_NAMES = {
    0: "Safety-Conscious Commuter",
    1: "Scenic Walker",
    2: "Efficient Minimalist",
}

BEHAVIOR_VECTORS = {
    0: np.array([0.8724,0.0514,0.8565,0.0517,0.7305,
                 0.6984,0.0187,0.9111,0.9306,0.7011], dtype=np.float32),
    1: np.array([0.2577,0.6485,0.1740,0.6020,0.1628,
                 0.2633,0.5536,0.2107,0.5857,0.1449], dtype=np.float32),
    2: np.array([0.7878,0.1303,0.7118,0.2033,0.4590,
                 0.6557,0.1498,0.6867,0.8446,0.5483], dtype=np.float32),
}

# ── Confirmed PPO hyperparameters ─────────────────────────────────────────────
PPO_HYPERPARAMS = dict(
    learning_rate = 1e-4,
    n_steps       = 4096,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.20,
    ent_coef      = 0.005,
    vf_coef       = 0.50,
    max_grad_norm = 0.50,
    verbose       = 1,
)

VARIANT_TOPK = {'v1s': 6, 'v2oracle': 10, 'v3': 6}


def make_personas_df():
    import pandas as pd
    # Column names must match traits_schema.py expectations exactly
    cols = ['Urgency','Crowd_Love','Crowd_Aversion','Walk_Love','Walk_Aversion',
            'Safety','Scenic_Love','Scenic_Aversion','Reliability','Cost']
    return pd.DataFrame([
        {'cluster': pid, **dict(zip(cols, vec.tolist()))}
        for pid, vec in BEHAVIOR_VECTORS.items()
    ])


def make_env(variant, persona_id, G, personas_df, oracle=None, eval_mode=False):
    """Create the correct environment for this variant."""
    sys.path.insert(0, str(MYCODE))
    top_k = VARIANT_TOPK.get(variant, 6)

    base_kw = dict(
        G=G, personas_df=personas_df, persona_id=persona_id,
        top_k=top_k, max_steps=150, time_budget_max=180.0,
        shaping_weight=50.0, bfs_cutoff=15,
        use_persona_budgets=False,
        near_goal_threshold=1 if eval_mode else 0,
    )

    if variant in ('v1s',):
        from phase2_environment_10d_clean import MultimodalRoutingEnv10D
        return MultimodalRoutingEnv10D(**base_kw)

    elif variant == 'v2oracle':
        from phase2_environment_10d_clean import MultimodalRoutingEnv10D
        from phase2_v2_oracle_environment import MultimodalRoutingEnvV2Oracle
        return MultimodalRoutingEnvV2Oracle(
            oracle_table=oracle, oracle_mode='inject', **base_kw)

    elif variant == 'v3':
        from phase2_v3_environment import RouteSelectionEnvV3
        return RouteSelectionEnvV3(
            G=G, personas_df=personas_df,
            persona_id=persona_id, k_routes=6, bfs_cutoff=15)

    raise ValueError(f"Unknown variant: {variant}")


def train(variant: str, persona_id: int,
          total_timesteps: int = 800_000, seed: int = 42):
    """Train one variant for one persona."""
    log.info("=" * 65)
    log.info(f"TRAIN  {variant.upper()}  P{persona_id}: {PERSONA_NAMES[persona_id]}")
    log.info(f"Steps: {total_timesteps:,}  |  seed={seed}")
    log.info("=" * 65)

    # Load graph
    graph_path = GRAPH_S if variant == 'v1s' else GRAPH_F
    log.info(f"Loading graph: {graph_path.name}")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    log.info(f"  {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges")

    # Load oracle for v2oracle
    oracle = None
    if variant == 'v2oracle' and ORACLE.exists():
        with open(ORACLE, 'rb') as f:
            oracle = pickle.load(f)['oracle']
        log.info(f"  Oracle: {len(oracle)} destinations loaded")

    personas_df = make_personas_df()

    # Environments
    train_env = Monitor(make_env(variant, persona_id, G, personas_df,
                                  oracle, eval_mode=False))
    eval_env  = Monitor(make_env(variant, persona_id, G, personas_df,
                                  oracle, eval_mode=True))
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    log.info(f"  Obs: {train_env.observation_space.shape}  "
             f"Actions: {train_env.action_space.n}  top_k={VARIANT_TOPK[variant]}")

    # Output directories
    out_dir  = RESULTS / f"variant_{variant}" / f"persona{persona_id}"
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = PPO("MlpPolicy", train_env, device="cpu",
                seed=seed, **PPO_HYPERPARAMS)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            CheckpointCallback(
                save_freq=50_000, save_path=str(out_dir),
                name_prefix=f"{variant}_p{persona_id}"),
            EvalCallback(
                eval_env, best_model_save_path=str(best_dir),
                eval_freq=25_000, n_eval_episodes=20, deterministic=True),
        ],
        progress_bar=True,
    )

    final = out_dir / f"{variant}_p{persona_id}_final.zip"
    model.save(str(final))
    log.info(f"Saved: {final}")
    log.info(f"Best model: {best_dir}/best_model.zip")
    return model


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Trainer")
    ap.add_argument('--variant', choices=['v1s','v2oracle','v3'])
    ap.add_argument('--persona', type=int, choices=[0,1,2])
    ap.add_argument('--all',    action='store_true',
                    help='Train all variants × all personas')
    ap.add_argument('--steps',  type=int, default=800_000)
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()

    if args.all:
        for variant in ['v1s', 'v2oracle', 'v3']:
            for pid in [0, 1, 2]:
                train(variant, pid, args.steps, args.seed)
    elif args.variant and args.persona is not None:
        train(args.variant, args.persona, args.steps, args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
