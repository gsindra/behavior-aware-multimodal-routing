"""
================================================================================
PHASE 2 — EVALUATOR
================================================================================
Evaluates all routing variants on shared OD pairs.
Produces results table, JS divergence, and per-persona mode shares.

Usage:
    python phase2_eval.py --all
    python phase2_eval.py --variants v2oracle v3 bwd
    python phase2_eval.py --variants v2oracle --n-episodes 200

Confirmed results (N=200, Wilson 95% CI):
    V1s   P0=0.5%  P1=0.0%  P2=0.0%     JS: P0vP1=0.003
    V2    P0=4.5%  P1=2.0%  P2=100.0%   JS: P0vP1=0.609  P1vP2=0.432
    V3    P0=100%  P1=100%  P2=100.0%   JS: P0vP1=0.111  P0vP2=0.129
    BWD   P0=100%  P1=100%  P2=100.0%   JS: P0vP1=0.075  P1vP2=0.196

Author: Indramuthu Sundaram — Phase 2, North Carolina A&T State University
================================================================================
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\gsind\North Carolina A&T State University"
               r"\Marwan Bikdash - Indra\Indra - Research Folder")
GRAPH_F = BASE / "Data"    / "Phase 3" / "G_phase3_rebuilt.pickle"
GRAPH_S = BASE / "Data"    / "Phase 3" / "G_phase3_small.pickle"
ORACLE  = BASE / "Results" / "Phase 2 Rerun Clean" / "Phase2_Oracle" / "phase2_oracle_500dest.pickle"
RESULTS = BASE / "Results" / "Phase 2 Rerun Clean"
MYCODE  = BASE / "MyCode"  / "Phase 2"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

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

VARIANT_TOPK = {'v1s': 6, 'v2oracle': 10, 'v3': 6}
VARIANT_GRAPH = {'v1s': 'small', 'v2oracle': 'full', 'v3': 'full', 'bwd': 'full'}


# ── Statistics ────────────────────────────────────────────────────────────────
def wilson_ci(pct: float, n: int) -> tuple:
    """Wilson score 95% CI. pct in [0,100]."""
    if n == 0: return (0.0, 0.0)
    p = pct / 100.0; z = 1.96
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return round(max(0,(c-m)*100), 1), round(min(100,(c+m)*100), 1)


def js_divergence(d1: dict, d2: dict) -> float:
    """Jensen-Shannon divergence between two mode distribution dicts."""
    modes = set(d1) | set(d2)
    t1 = sum(d1.values()) or 1; t2 = sum(d2.values()) or 1
    p = np.array([d1.get(m,0)/t1 for m in modes])
    q = np.array([d2.get(m,0)/t2 for m in modes])
    m = 0.5*(p+q)
    def kl(a, b):
        mask = (a>0)&(b>0)
        return np.sum(a[mask]*np.log(a[mask]/b[mask]))
    return round(float(0.5*kl(p,m)+0.5*kl(q,m)), 4)


# ── OD pair generation ────────────────────────────────────────────────────────
def generate_od_pairs(G, n: int, seed: int = 999,
                      bfs_cutoff: int = 15,
                      oracle_dests: list = None) -> list:
    """
    Shared OD pairs for fair cross-persona comparison.
    Street nodes only. BFS hop-count reachability check.
    """
    import networkx as nx
    street = [nd for nd, d in G.nodes(data=True)
              if d.get('node_type') != 'transit_stop']
    pool   = oracle_dests if oracle_dests else street
    rng    = np.random.default_rng(seed)
    pairs  = []
    att    = 0
    while len(pairs) < n and att < n * 50:
        att += 1
        o = street[int(rng.integers(0, len(street)))]
        d = pool[int(rng.integers(0, len(pool)))]
        if o == d: continue
        try:
            h = nx.shortest_path_length(G, o, d)
            if 3 <= h <= bfs_cutoff:
                pairs.append((o, d))
        except Exception:
            pass
    log.info(f"  Generated {len(pairs)} OD pairs")
    return pairs


# ── Model finder ──────────────────────────────────────────────────────────────
def find_model(variant: str, persona_id: int) -> Path | None:
    """Locate trained model for this variant and persona."""
    v2o_dirs = [
        RESULTS / 'variant_v2_oracle_final'  / f'persona{persona_id}',
        RESULTS / 'variant_v2_oracle_strict' / f'persona{persona_id}',
        RESULTS / 'variant_v2_oracle'        / f'persona{persona_id}',
    ]
    v2o_dir = next((d for d in v2o_dirs if d.exists()), v2o_dirs[0])

    base_dirs = {
        'v1s':      RESULTS / 'variant_v1s'  / f'persona{persona_id}',
        'v2oracle': v2o_dir,
        'v3':       RESULTS / 'variant_v3'   / f'persona{persona_id}',
    }
    d = base_dirs.get(variant)
    if not d: return None

    if variant in ('v2oracle', 'v3'):
        search = [*sorted(d.glob('*final*.zip')),
                  d / 'best' / 'best_model.zip',
                  *sorted(d.glob('*.zip'))]
    else:
        search = [d / 'best' / 'best_model.zip',
                  *sorted(d.glob('*final*.zip')),
                  *sorted(d.glob('*.zip'))]
    for p in search:
        if p.exists(): return p
    return None


# ── Environment factory ───────────────────────────────────────────────────────
def make_env(variant, persona_id, G, personas_df, oracle=None):
    sys.path.insert(0, str(MYCODE))
    top_k = VARIANT_TOPK.get(variant, 6)
    base_kw = dict(
        G=G, personas_df=personas_df, persona_id=persona_id,
        top_k=top_k, max_steps=150, time_budget_max=180.0,
        shaping_weight=50.0, bfs_cutoff=15,
        use_persona_budgets=False, near_goal_threshold=0,
    )
    if variant == 'v1s':
        from phase2_environment_10d_clean import MultimodalRoutingEnv10D
        return MultimodalRoutingEnv10D(**base_kw)
    elif variant == 'v2oracle':
        from phase2_v2_oracle_environment import MultimodalRoutingEnvV2Oracle
        return MultimodalRoutingEnvV2Oracle(
            oracle_table=oracle, oracle_mode='inject', **base_kw)
    elif variant == 'v3':
        from phase2_v3_environment import RouteSelectionEnvV3
        return RouteSelectionEnvV3(G=G, personas_df=personas_df,
            persona_id=persona_id, k_routes=6, bfs_cutoff=15)
    raise ValueError(f"Unknown variant: {variant}")


def make_personas_df():
    import pandas as pd
    cols = ['Urgency','Crowd_Love','Crowd_Aversion','Walk_Love','Walk_Aversion',
            'Safety_Sensitivity','Scenic_Love','Scenic_Aversion',
            'Reliability_Expectation','Cost_Sensitivity']
    return pd.DataFrame([
        {'cluster': pid, **dict(zip(cols, vec.tolist()))}
        for pid, vec in BEHAVIOR_VECTORS.items()
    ])


# ── PPO variant evaluator ─────────────────────────────────────────────────────
def evaluate_ppo(variant: str, G, personas_df, od_pairs: list,
                 oracle=None, n_ep: int = 200) -> tuple:
    from stable_baselines3 import PPO
    results = {}; mode_dists = {}

    for pid in [0, 1, 2]:
        mp = find_model(variant, pid)
        if mp is None:
            log.warning(f"  {variant} P{pid}: model not found — skipping")
            continue
        log.info(f"\n  {variant.upper()} P{pid} ({PERSONA_NAMES[pid]})")
        log.info(f"  Model: {mp.name}")

        model = PPO.load(str(mp), device='cpu')
        env   = make_env(variant, pid, G, personas_df, oracle)
        rows  = []

        for ep, (o, d) in enumerate(od_pairs[:n_ep]):
            try:
                obs, _ = env.reset(options={'trip':{'origin':o,'destination':d}})
            except Exception:
                obs, _ = env.reset()

            done = trunc = False; ep_rew = 0
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, done, trunc, info = env.step(int(action))
                ep_rew += rew

            rows.append({
                'episode':         ep + 1,
                'variant':         variant,
                'persona_id':      pid,
                'persona_name':    PERSONA_NAMES[pid],
                'reward':          round(ep_rew, 2),
                'steps':           info.get('num_steps', 0),
                'success_strict':  info.get('success_strict',
                                   int(info.get('reached_destination', 0))),
                'success_relaxed': info.get('success_relaxed',
                                   info.get('success', 0)),
                'modes_used':      info.get('modes_used', ''),
                'timeout_reason':  info.get('timeout_reason', ''),
            })

            if (ep + 1) % 50 == 0:
                s = np.mean([r['success_strict'] for r in rows[-50:]]) * 100
                log.info(f"    Ep {ep+1}/{n_ep} | strict={s:.0f}%")

        df  = pd.DataFrame(rows)
        n   = len(df)
        pct = df['success_strict'].mean() * 100
        ci  = wilson_ci(pct, n)

        all_modes = []
        for seq in df['modes_used'].fillna(''):
            all_modes.extend([m for m in seq.split(',') if m])
        mc  = Counter(all_modes)
        tot = sum(mc.values()) or 1
        mode_dists[pid] = mc

        log.info(f"    Strict: {pct:.1f}% [{ci[0]},{ci[1]}]  "
                 + "  ".join(f"{m}={100*c/tot:.0f}%"
                 for m,c in sorted(mc.items(),key=lambda x:-x[1])[:4]))

        out = RESULTS / f"master_eval_{variant}_p{pid}_{n_ep}ep.csv"
        df.to_csv(out, index=False)
        results[pid] = {'strict': pct, 'ci': ci, 'df': df}

    js = {(i,j): js_divergence(mode_dists[i], mode_dists[j])
          for i,j in [(0,1),(0,2),(1,2)]
          if i in mode_dists and j in mode_dists}
    return results, mode_dists, js


# ── BWD evaluator ─────────────────────────────────────────────────────────────
def evaluate_bwd(G, od_pairs: list, n_ep: int = 200) -> tuple:
    """Behavior-Weighted Dijkstra — no PPO, deterministic routing."""
    import networkx as nx

    MODE_ATTRS = {
        'walk':           {'safety':0.6,'scenic':0.9,'reliability':0.7,'crowd':0.3,'cost':0.0},
        'bike':           {'safety':0.5,'scenic':0.8,'reliability':0.6,'crowd':0.2,'cost':0.0},
        'bus':            {'safety':0.7,'scenic':0.6,'reliability':0.5,'crowd':0.8,'cost':0.3},
        'subway':         {'safety':0.8,'scenic':0.3,'reliability':0.9,'crowd':0.9,'cost':0.3},
        'car':            {'safety':0.9,'scenic':0.5,'reliability':0.85,'crowd':0.1,'cost':0.8},
        'ride-hail':      {'safety':0.9,'scenic':0.4,'reliability':0.8,'crowd':0.1,'cost':1.0},
        'transit_access': {'safety':0.7,'scenic':0.6,'reliability':0.7,'crowd':0.4,'cost':0.0},
    }

    def score(mode, t, b):
        ma = MODE_ATTRS.get(mode, MODE_ATTRS['walk'])
        return (t*(0.3+2.0*b[0]) - ma['safety']*b[5]*3.0
                + ma['cost']*b[9]*5.0 + ma['crowd']*(b[2]-b[1])
                - ma['scenic']*b[6]*2.0 - ma['reliability']*b[8])

    results = {}; mode_dists = {}

    for pid in [0, 1, 2]:
        b         = BEHAVIOR_VECTORS[pid]
        is_safety = float(b[5]) > 0.65 and float(b[6]) < 0.25
        is_scenic  = float(b[6]) > 0.5  and float(b[0]) < 0.4

        def ew(u, v, d, _b=b, _s=is_safety, _sc=is_scenic):
            if isinstance(d, dict) and 0 in d: d = d[0]
            m = d.get('mode', 'walk')
            if (_s or _sc) and m in ('car','ride-hail'): return 1e9
            t = float(d.get('travel_time', d.get('time', 60.0))) / 60.0
            return max(score(m, t, _b), 0.001)

        log.info(f"\n  BWD P{pid} ({PERSONA_NAMES[pid]})")
        rows = []; all_modes = []

        for ep, (o, d) in enumerate(od_pairs[:n_ep]):
            try:
                path    = nx.dijkstra_path(G, o, d, weight=ew)
                success = 1 if path[-1] == d else 0
            except Exception:
                path = [o]; success = 0

            modes = []
            for i in range(len(path)-1):
                ed = G.get_edge_data(path[i], path[i+1])
                if ed:
                    if isinstance(ed,dict) and 0 in ed: ed=ed[0]
                    modes.append(ed.get('mode','walk'))
            all_modes.extend(modes)
            rows.append({
                'episode': ep+1, 'variant': 'bwd', 'persona_id': pid,
                'persona_name': PERSONA_NAMES[pid],
                'success_strict': success, 'success_relaxed': success,
                'steps': len(path)-1, 'modes_used': ','.join(modes),
                'reward': 0, 'travel_time': 0,
                'timeout_reason': '' if success else 'no_path',
            })

        df  = pd.DataFrame(rows)
        pct = df['success_strict'].mean() * 100
        ci  = wilson_ci(pct, len(df))
        mc  = Counter(all_modes)
        tot = sum(mc.values()) or 1
        mode_dists[pid] = mc

        log.info(f"    Strict: {pct:.1f}% [{ci[0]},{ci[1]}]  "
                 + "  ".join(f"{m}={100*c/tot:.0f}%"
                 for m,c in sorted(mc.items(),key=lambda x:-x[1])[:4]))

        out = RESULTS / f"master_eval_bwd_p{pid}_{n_ep}ep.csv"
        df.to_csv(out, index=False)
        results[pid]    = {'strict': pct, 'ci': ci, 'df': df}
        mode_dists[pid] = mc

    js = {(i,j): js_divergence(mode_dists[i], mode_dists[j])
          for i,j in [(0,1),(0,2),(1,2)]}
    return results, mode_dists, js


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(all_results: dict):
    print("\n" + "="*75)
    print("PHASE 2 RESULTS (N=200, Wilson 95% CI)")
    print("="*75)
    labels = {'v1s':'V1s Pure PPO','v2oracle':'V2 Oracle BC-PPO',
              'v3':'V3 Route Selection','bwd':'BWD Dijkstra'}
    for vkey, vname in labels.items():
        if vkey not in all_results: continue
        res = all_results[vkey]['res']
        js  = all_results[vkey]['js']
        for pid, r in res.items():
            lbl = vname if pid == 0 else ''
            print(f"  {lbl:<22} P{pid} {PERSONA_NAMES[pid]:<28} "
                  f"{r['strict']:>6.1f}%  [{r['ci'][0]},{r['ci'][1]}]")
        if js:
            print(f"  {'':22} JS: P0vP1={js.get((0,1),0):.4f}  "
                  f"P0vP2={js.get((0,2),0):.4f}  "
                  f"P1vP2={js.get((1,2),0):.4f}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Phase 2 Evaluator")
    ap.add_argument('--variants', nargs='+',
                    choices=['v1s','v2oracle','v3','bwd'],
                    default=['v1s','v2oracle','v3','bwd'])
    ap.add_argument('--all',        action='store_true')
    ap.add_argument('--n-episodes', type=int, default=200)
    ap.add_argument('--seed',       type=int, default=999)
    args = ap.parse_args()

    variants = ['v1s','v2oracle','v3','bwd'] if args.all else args.variants

    log.info("Loading graphs...")
    with open(GRAPH_F,'rb') as f: G_full  = pickle.load(f)
    with open(GRAPH_S,'rb') as f: G_small = pickle.load(f)
    log.info(f"Full: {G_full.number_of_nodes():,}  Small: {G_small.number_of_nodes():,}")

    oracle = None
    if 'v2oracle' in variants and ORACLE.exists():
        with open(ORACLE,'rb') as f: oracle = pickle.load(f)['oracle']
        log.info(f"Oracle: {len(oracle)} destinations")

    personas_df  = make_personas_df()
    oracle_dests = list(oracle.keys()) if oracle else None

    od_full  = generate_od_pairs(G_full,  args.n_episodes, args.seed,
                                  bfs_cutoff=15, oracle_dests=oracle_dests)
    od_small = generate_od_pairs(G_small, args.n_episodes, args.seed,
                                  bfs_cutoff=25)

    all_results = {}
    for variant in variants:
        log.info(f"\n{'='*60}\nVARIANT: {variant.upper()}\n{'='*60}")
        G  = G_small if variant == 'v1s' else G_full
        od = od_small if variant == 'v1s' else od_full

        if variant == 'bwd':
            res, mds, js = evaluate_bwd(G, od, args.n_episodes)
        else:
            res, mds, js = evaluate_ppo(variant, G, personas_df, od,
                                         oracle, args.n_episodes)
        all_results[variant] = {'res': res, 'mds': mds, 'js': js}

    print_summary(all_results)

    # Save master CSV
    rows = []
    for vkey, data in all_results.items():
        for pid, r in data['res'].items():
            mc  = data['mds'].get(pid, {})
            tot = sum(mc.values()) or 1
            rows.append({
                'variant':    vkey,
                'persona_id': pid,
                'persona':    PERSONA_NAMES[pid],
                'strict_%':   round(r['strict'], 1),
                'ci_lo':      r['ci'][0],
                'ci_hi':      r['ci'][1],
                'walk_%':     round(100*mc.get('walk',0)/tot, 1),
                'transit_%':  round(100*(mc.get('bus',0)+mc.get('subway',0))/tot, 1),
                'access_%':   round(100*mc.get('transit_access',0)/tot, 1),
                'car_%':      round(100*(mc.get('car',0)+mc.get('ride-hail',0))/tot, 1),
                'JS_P0P1':    data['js'].get((0,1),'') if pid==0 else '',
                'JS_P0P2':    data['js'].get((0,2),'') if pid==0 else '',
                'JS_P1P2':    data['js'].get((1,2),'') if pid==1 else '',
            })
    out = RESULTS / f"phase2_master_eval_summary_{args.n_episodes}ep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"Summary saved: {out.name}")


if __name__ == "__main__":
    main()
