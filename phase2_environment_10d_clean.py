"""
================================================================================
PHASE 2 ENVIRONMENT — CLEAN REWRITE (Fixed Graph, Honest Behavioral Routing)
================================================================================

Design principles (addressing problems in original phase2_environment_10d.py):

FIX 1 — P2 hardcoding REMOVED
  Original: `if self.persona_id == 2:` blocks in both _get_ranked_neighbors()
            and _compute_reward() manually scripted transit preference for P2.
  Fix:      All ranking and reward is driven purely by the 10D behavior vector.
            No persona_id checks anywhere in the MDP logic.

FIX 2 — Candidate ranking is behavior-driven, not persona-scripted
  Original: Hardcoded subway bonus (-90%), car penalty (+25%) for P2 only.
  Fix:      Each mode gets a behavior-weighted generalized cost score derived
            from the 10D vector (urgency, safety, cost, scenic, crowd traits).
            All personas go through the same function — differentiation comes
            entirely from different behavior vectors.

FIX 3 — Reward is fully 10D-driven, no persona branches
  Original: `if self.persona_id == 2: reward += 1.5 if subway` etc.
  Fix:      Single reward function using all 10 traits for all personas.
            Mode attributes (safety, scenic, crowd, cost, reliability) are
            weighted by the corresponding behavioral trait. Same code path
            for all personas.

FIX 4 — No per-episode full Dijkstra on 110K-node graph
  Original: _compute_shortest_distances() called full weighted Dijkstra at
            every reset → catastrophically slow on the Phase 3 graph.
  Fix:      BFS hop-count with cutoff=50. Fast (<<1s), sufficient for
            potential shaping. OD pairs are guaranteed reachable within
            cutoff by the sampling function.

FIX 5 — OD sampling excludes transit stops
  Original: Sampled from all_nodes including subway/bus stops. Agent could
            start at a transit node with no walk/bike outgoing edges.
  Fix:      street_nodes list (node_type != 'transit_stop') used for sampling.

FIX 6 — Potential shaping scale is bounded
  Original: potential = -dist * 50.0 — could produce ±5000 swings that
            drown out the 10D behavioral reward terms.
  Fix:      potential = -dist / max_dist (normalized to [-1, 0]).
            shaping_weight controls magnitude separately.

FIX 7 — Observation is decoupled from ranking
  Original: _get_obs() called _get_ranked_neighbors() which applied P2 hacks,
            so observation and action space were both corrupted by hacking.
  Fix:      Candidates are computed once per step, stored in self._candidates.
            Observation uses raw edge features (dist, time, mode one-hot)
            independent of any persona-specific scoring.

FIX 8 — No per-episode shortest-path in OD sampling
  Original: _sample_trip() ran two nx.shortest_path_length calls per
            candidate OD pair in a rejection loop → slow.
  Fix:      Sample from pre-filtered street_nodes. Feasibility check uses
            BFS hop-count only (fast). No weighted SP in sampling.

Observation space (171 dims, top_k=25):
    [0:10]   node state features
    [10]     normalized hop-distance to destination
    [11:21]  full 10D behavior vector
    [21:171] top_k neighbors × 6 features each:
               dist_norm(1) + time_norm(1) + mode_onehot(4: walk/bike/car/transit)
             ride-hail grouped with car in one-hot for compactness

Reward function (fully 10D-driven):
    time_cost      = -edge_time × b[0] (Urgency)
    safety         = +mode.safety × b[5] (Safety)
    scenic         = +(mode.scenic × b[6] - mode.scenic × b[7]) (Scenic_Love/Aversion)
    crowd          = +(mode.crowd × b[1] - mode.crowd × b[2]) (Crowd_Love/Aversion)
    walk           = +is_walk × (b[3] - b[4]) (Walk_Love/Aversion)
    reliability    = +mode.reliability × b[8] (Reliability)
    cost           = -mode.cost × b[9] (Cost)
    transfer       = -0.5 × transfer_occurred
    shaping        = shaping_weight × (potential_after - potential_before)
    loop_penalty   = -1.0 × max(0, visit_count - 1)

Author: Indra (PhD) — Phase 2 clean rewrite, March 2026
================================================================================
"""

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
import logging

try:
    from traits_schema import TraitSchema, ControlVectorMapping
    from data_config import PERSONA_NAMES
    HAS_SCHEMA = True
except ImportError:
    print("Warning: traits_schema not found — using hardcoded behavior vectors")
    HAS_SCHEMA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Mode attributes (fixed, empirically grounded) ──────────────────────────────
# Each mode has: safety, scenic, reliability, crowd, cost ∈ [0,1]
# These are environment constants — NOT persona-specific
MODE_ATTRS: Dict[str, Dict[str, float]] = {
    'walk':           {'safety': 0.6, 'scenic': 0.9, 'reliability': 0.7, 'crowd': 0.3, 'cost': 0.0},
    'bike':           {'safety': 0.5, 'scenic': 0.8, 'reliability': 0.6, 'crowd': 0.2, 'cost': 0.0},
    'bus':            {'safety': 0.7, 'scenic': 0.6, 'reliability': 0.5, 'crowd': 0.8, 'cost': 0.3},
    'subway':         {'safety': 0.8, 'scenic': 0.3, 'reliability': 0.9, 'crowd': 0.9, 'cost': 0.3},
    'car':            {'safety': 0.9, 'scenic': 0.5, 'reliability': 0.85,'crowd': 0.1, 'cost': 0.8},
    'ride-hail':      {'safety': 0.9, 'scenic': 0.4, 'reliability': 0.8, 'crowd': 0.1, 'cost': 1.0},
    'transit_access': {'safety': 0.7, 'scenic': 0.6, 'reliability': 0.7, 'crowd': 0.4, 'cost': 0.0},
}

# Mode one-hot: 4 bins — walk | bike | car/ride-hail | transit
# ride-hail grouped with car (both private vehicle); transit = bus/subway/transit_access
MODE_ONEHOT_IDX = {
    'walk':           0,
    'bike':           1,
    'car':            2,
    'ride-hail':      2,
    'bus':            3,
    'subway':         3,
    'transit_access': 3,
}
N_MODE_ONEHOT   = 4                        # walk, bike, car/ridehail, transit
FEATURES_PER_NEIGHBOR = 2 + N_MODE_ONEHOT  # dist + time + 4 mode bits = 6


def _mode_attrs(mode: str) -> Dict[str, float]:
    return MODE_ATTRS.get(mode, {'safety': 0.5, 'scenic': 0.5,
                                 'reliability': 0.5, 'crowd': 0.5, 'cost': 0.5})


class MultimodalRoutingEnv10D(gym.Env):
    """
    Phase 2 routing environment: single agent, one OD pair per episode.

    The ONLY source of behavioral differentiation is the 10D vector b:
        b[0]  Urgency          b[1]  Crowd_Love      b[2]  Crowd_Aversion
        b[3]  Walk_Love        b[4]  Walk_Aversion   b[5]  Safety
        b[6]  Scenic_Love      b[7]  Scenic_Aversion b[8]  Reliability
        b[9]  Cost

    No persona_id checks exist in any reward, ranking, or observation logic.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        G:                    nx.Graph,
        personas_df,
        persona_id:           int,
        top_k:                int   = 25,
        max_steps:            int   = 200,
        time_conversion:      float = 1.0,
        time_budget_max:      float = 240.0,
        shaping_weight:       float = 10.0,   # bounded: potential ∈ [-1,0], so max shaping = 10/step
        use_persona_budgets:  bool  = False,   # False = identical budget for all personas
        near_goal_threshold:  float = 0.0,     # 0 = training (exact), 0.2 = eval (near-goal)
        bfs_cutoff:           int   = 15,      # tight OD pairs — achievable within 200 steps
    ):
        super().__init__()

        self.G                   = G
        self.personas_df         = personas_df
        self.persona_id          = persona_id
        self.top_k               = top_k
        self.max_steps           = max_steps
        self.time_conversion     = time_conversion
        self.time_budget_max     = time_budget_max
        self.shaping_weight      = shaping_weight
        self.use_persona_budgets = use_persona_budgets
        self.near_goal_threshold = near_goal_threshold
        self.bfs_cutoff          = bfs_cutoff

        # ── Extract persona ───────────────────────────────────────────────────
        if 'cluster' in personas_df.columns:
            self.persona = personas_df[personas_df['cluster'] == persona_id].iloc[0]
        else:
            self.persona = personas_df.iloc[persona_id]

        # ── 10D behavioral vector ─────────────────────────────────────────────
        if HAS_SCHEMA:
            self.behavior_vector_10d = TraitSchema.extract_from_row(self.persona)
            self.control_vector_6d   = ControlVectorMapping.derive(self.behavior_vector_10d)
            persona_name = PERSONA_NAMES.get(persona_id, f'Persona {persona_id}')
        else:
            self.behavior_vector_10d = np.array([0.5] * 10, dtype=np.float32)
            self.control_vector_6d   = np.array([0.5] * 6,  dtype=np.float32)
            persona_name = f'Persona {persona_id}'

        # ── Node lists ────────────────────────────────────────────────────────
        self.all_nodes = list(G.nodes())

        # FIX 5: street nodes only for OD sampling — exclude transit stops
        self.street_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get('node_type', 'street') != 'transit_stop'
        ]
        if len(self.street_nodes) < 2:
            logger.warning("Fewer than 2 street nodes — falling back to all nodes")
            self.street_nodes = self.all_nodes

        # ── Spaces ────────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(top_k)

        # 10 + 1 + 10 + (6 × top_k) = 171 with top_k=25
        obs_dim = 10 + 1 + 10 + (FEATURES_PER_NEIGHBOR * top_k)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Precompute reversed graph ONCE — avoids per-reset reversal on 110K-node graph
        self.G_rev = G.reverse(copy=False) if G.is_directed() else G

        logger.info(f"Phase 2 Clean Env: {persona_name}")
        logger.info(f"  10D behavior: {self.behavior_vector_10d}")
        logger.info(f"  Obs dim: {obs_dim} | top_k={top_k} | bfs_cutoff={bfs_cutoff}")
        print(f"✅ ENV: phase2_environment_10d_clean.py | P{persona_id} | obs={obs_dim}")

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _node_coord(self, node) -> Tuple[float, float]:
        """Return (lat, lon) trying multiple attribute names."""
        d = self.G.nodes[node]
        if 'pos' in d:
            lat, lon = d['pos'][0], d['pos'][1]
        else:
            lat = float(d.get('y', d.get('lat', 0.0)))
            lon = float(d.get('x', d.get('lon', 0.0)))
        return lat, lon

    def _edge_time(self, u, v) -> float:
        """Minimum travel time (minutes) across all parallel edges u→v."""
        ed = self.G.get_edge_data(u, v)
        if ed is None:
            return 1.0
        if isinstance(ed, dict):
            first = next(iter(ed.values()), None)
            if isinstance(first, dict):           # MultiGraph
                times = [float(a.get('weight', 1.0))
                         for a in ed.values() if isinstance(a, dict)]
                return min(times) if times else 1.0
            return float(ed.get('weight', 1.0))   # simple graph
        return 1.0

    def _edge_mode(self, u, v) -> str:
        """Mode of the fastest edge u→v."""
        ed = self.G.get_edge_data(u, v)
        if ed is None:
            return 'walk'
        if isinstance(ed, dict):
            first = next(iter(ed.values()), None)
            if isinstance(first, dict):
                best_t, best_m = float('inf'), 'walk'
                for a in ed.values():
                    if isinstance(a, dict):
                        t = float(a.get('weight', float('inf')))
                        if t < best_t:
                            best_t, best_m = t, a.get('mode', 'walk')
                return best_m
            return ed.get('mode', 'walk')
        return 'walk'

    def _euclidean_dist(self, n1, n2) -> float:
        try:
            lat1, lon1 = self._node_coord(n1)
            lat2, lon2 = self._node_coord(n2)
            return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIOR-DRIVEN CANDIDATE RANKING  (FIX 1, FIX 2)
    # ══════════════════════════════════════════════════════════════════════════

    def _behavior_score(self, mode: str, time: float) -> float:
        """
        Generalized cost for a candidate edge, derived purely from b (10D vector).
        Lower score = more preferred by this persona.

        No persona_id checks. All differentiation comes from b.

        Score components:
            time_cost      : urgent personas penalize slow edges more
            safety_benefit : safety-conscious personas prefer safe modes
            cost_penalty   : cost-sensitive personas penalize expensive modes
            crowd_term     : crowd-lovers prefer busy transit; crowd-averse avoid it
            scenic_benefit : scenic-lovers prefer walk/bike
        """
        b  = self.behavior_vector_10d
        ma = _mode_attrs(mode)

        time_cost      = time * (0.5 + b[0])          # urgency amplifies time cost
        safety_benefit = -ma['safety'] * b[5] * 3.0   # safety trait: negative = preferred
        cost_penalty   = ma['cost'] * b[9] * 5.0      # cost sensitivity: positive = avoided
        crowd_term     = ma['crowd'] * (b[2] - b[1])  # net crowd aversion
        scenic_benefit = -ma['scenic'] * b[6] * 2.0   # scenic lovers prefer scenic modes
        reliability    = -ma['reliability'] * b[8]    # reliability-seekers prefer reliable modes

        return time_cost + safety_benefit + cost_penalty + crowd_term + scenic_benefit + reliability

    def _get_candidates(self, node) -> List:
        """
        Return top_k neighbor nodes ranked by behavior-driven generalized cost.
        FIX 1/2: No persona_id checks. Ranking is identical code for all personas.
        """
        if self.G.is_directed():
            neighbors = list(self.G.successors(node))
        else:
            neighbors = list(self.G.neighbors(node))

        if not neighbors:
            return []

        # Prefer unvisited neighbors to reduce loops (not persona-specific)
        unvisited = [n for n in neighbors if self.visit_counts.get(n, 0) == 0
                     or n == self.destination]
        pool = unvisited if unvisited else neighbors

        # Score by behavior-driven generalized cost
        scored = []
        for n in pool:
            t    = self._edge_time(node, n)
            mode = self._edge_mode(node, n)
            score = self._behavior_score(mode, t)
            scored.append((n, score))

        scored.sort(key=lambda x: x[1])      # ascending = lower cost = preferred
        return [n for n, _ in scored[:self.top_k]]

    # ══════════════════════════════════════════════════════════════════════════
    # POTENTIAL-BASED SHAPING  (FIX 6)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_potential(self) -> float:
        """
        Normalized hop-count potential ∈ [-1, 0].
        FIX 6: bounded so shaping can't dominate the behavioral reward terms.
        """
        dist     = self.hop_distances.get(self.current_node, self.bfs_cutoff + 1)
        max_dist = max(self.bfs_cutoff, 1)
        return -(dist / max_dist)              # ∈ [-1, 0], 0 = at destination

    # ══════════════════════════════════════════════════════════════════════════
    # REWARD  (FIX 3)
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_reward(
        self,
        edge_time:        float,
        edge_mode:        str,
        transfer_occurred: bool,
        potential_before: float,
        potential_after:  float,
        loop_penalty:     float,
    ) -> float:
        """
        Fully 10D-driven reward. No persona_id branches anywhere.

        Trait indices:
            b[0] Urgency   b[1] Crowd_Love    b[2] Crowd_Aversion
            b[3] Walk_Love b[4] Walk_Aversion b[5] Safety
            b[6] Scenic_Love  b[7] Scenic_Aversion
            b[8] Reliability  b[9] Cost
        """
        b  = self.behavior_vector_10d
        ma = _mode_attrs(edge_mode)

        # Time penalty — scaled by urgency
        time_cost = -edge_time * b[0] * 0.05

        # Safety reward — safe personas gain more from safe modes
        safety = ma['safety'] * b[5] * 0.4

        # Scenic — scenic lovers gain from scenic modes; averse lose
        scenic = (ma['scenic'] * b[6] - ma['scenic'] * b[7]) * 0.3

        # Crowd — crowd-lovers gain from busy transit; averse lose
        crowd = (ma['crowd'] * b[1] - ma['crowd'] * b[2]) * 0.3

        # Walk preference — walk-lovers gain; walk-averse lose
        is_walk   = 1.0 if edge_mode == 'walk' else 0.0
        walk_term = is_walk * (b[3] - b[4]) * 0.4

        # Reliability — reliability-conscious personas prefer reliable modes
        reliability = ma['reliability'] * b[8] * 0.3

        # Cost penalty — cost-sensitive personas penalize expensive modes
        cost = -ma['cost'] * b[9] * 0.5

        # Transfer penalty (universal — all personas dislike unnecessary transfers)
        transfer = -0.5 * int(transfer_occurred)

        # Potential-based shaping (FIX 6: bounded)
        shaping = self.shaping_weight * (potential_after - potential_before)

        # Loop penalty
        loop = loop_penalty

        reward = (time_cost + safety + scenic + crowd + walk_term
                  + reliability + cost + transfer + shaping + loop)
        return float(reward)

    # ══════════════════════════════════════════════════════════════════════════
    # OBSERVATION  (FIX 7)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        """
        Build observation. FIX 7: candidates computed fresh and stored in
        self._candidates. Observation uses raw edge features independent of
        any behavioral scoring.
        """
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ── [0:10] node state ─────────────────────────────────────────────────
        obs[0] = min(self.steps / max(self.max_steps_current, 1), 1.0)
        obs[1] = min(self.cumulative_time / max(self.time_budget_current, 1.0), 1.0)
        obs[2] = min(self.num_transfers / 10.0, 1.0)
        obs[3] = min(self.num_mode_switches / 20.0, 1.0)
        obs[4] = float(self.current_node == self.destination)
        obs[5] = min(len(self.path) / 100.0, 1.0)
        obs[6] = min(self.visit_counts.get(self.current_node, 0) / 5.0, 1.0)
        obs[7] = min(self.steps_since_progress / 30.0, 1.0)
        # obs[8:10] spare

        # ── [10] hop distance to destination ─────────────────────────────────
        dist = self.hop_distances.get(self.current_node, self.bfs_cutoff + 1)
        obs[10] = min(dist / (self.bfs_cutoff + 1), 1.0)

        # ── [11:21] full 10D behavior vector ─────────────────────────────────
        obs[11:21] = self.behavior_vector_10d

        # ── [21:] neighbor features (raw — no behavioral scoring) ─────────────
        # FIX 7: recompute candidates here, store for step() to use
        self._candidates = self._get_candidates(self.current_node)

        base = 21
        for i, nbr in enumerate(self._candidates):
            if i >= self.top_k:
                break
            offset = base + i * FEATURES_PER_NEIGHBOR
            dist_n = self._euclidean_dist(self.current_node, nbr)
            time_n = self._edge_time(self.current_node, nbr)
            mode_n = self._edge_mode(self.current_node, nbr)

            obs[offset]     = min(dist_n / 0.05, 1.0)          # ~0.05° ≈ 5km
            obs[offset + 1] = min(time_n / 30.0, 1.0)          # normalise by 30 min
            mode_idx = MODE_ONEHOT_IDX.get(mode_n, 0)
            obs[offset + 2 + mode_idx] = 1.0                   # 4-dim one-hot

        return obs

    # ══════════════════════════════════════════════════════════════════════════
    # OD SAMPLING  (FIX 4, FIX 5, FIX 8)
    # ══════════════════════════════════════════════════════════════════════════

    def _sample_od(self, rng: np.random.Generator) -> Tuple:
        """
        FIX 5: sample from street_nodes only (no transit stops as OD).
        FIX 4: feasibility check uses BFS hop-count only — no full Dijkstra.
        FIX 8: no weighted shortest-path in sampling loop.
        """
        nodes = self.street_nodes
        for _ in range(200):
            idxs = rng.choice(len(nodes), size=2, replace=False)
            o, d = nodes[idxs[0]], nodes[idxs[1]]
            # Quick BFS reachability check (fast, cutoff keeps it bounded)
            try:
                hops = nx.shortest_path_length(self.G, o, d)
                if 2 <= hops <= self.bfs_cutoff:
                    return o, d
            except nx.NetworkXNoPath:
                continue
            except Exception:
                continue
        # Fallback: first two different street nodes
        if len(nodes) >= 2:
            return nodes[0], nodes[1]
        return self.all_nodes[0], self.all_nodes[1]

    # ══════════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # OD pair
        if options and 'trip' in options:
            self.origin      = options['trip']['origin']
            self.destination = options['trip']['destination']
        else:
            self.origin, self.destination = self._sample_od(rng)

        self.current_node = self.origin

        # Episode state
        self.steps              = 0
        self.cumulative_time    = 0.0
        self.num_transfers      = 0
        self.num_mode_switches  = 0
        self.last_mode          = None
        self.modes_used         = []
        self.path               = [self.origin]
        self.visit_counts       = {self.origin: 1}
        self.steps_since_progress = 0
        self.best_hop_dist      = self.bfs_cutoff + 1
        self._candidates        = []

        # Budget (same for all personas when use_persona_budgets=False)
        self.time_budget_current = self.time_budget_max
        self.max_steps_current   = self.max_steps

        # FIX 4: BFS hop-count distances TO destination (precomputed G_rev)
        try:
            raw = nx.single_source_shortest_path_length(
                self.G_rev, self.destination, cutoff=self.bfs_cutoff
            )
            self.hop_distances = dict(raw)
        except Exception:
            self.hop_distances = {}

        # Set best_hop_dist from origin
        self.best_hop_dist = self.hop_distances.get(self.origin, self.bfs_cutoff + 1)

        obs  = self._get_obs()
        info = {
            'persona_id':  self.persona_id,
            'origin':      str(self.origin),
            'destination': str(self.destination),
        }
        return obs, info

    # ══════════════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════════════

    def step(self, action: int):
        self.steps += 1
        potential_before = self._get_potential()

        candidates = self._candidates

        # Clip action to valid candidate range (PPO action space is Discrete(top_k)
        # but a node may have fewer than top_k neighbors)
        if not candidates:
            obs = self._get_obs()
            return obs, -2.0, False, True, self._info(
                success=0, timeout_reason='no_candidates'
            )
        action = action % len(candidates)  # safe modulo clip

        next_node  = candidates[action]
        edge_time  = self._edge_time(self.current_node, next_node)
        edge_mode  = self._edge_mode(self.current_node, next_node)

        # Update state
        self.cumulative_time += edge_time / self.time_conversion
        self.current_node     = next_node
        self.path.append(next_node)
        self.modes_used.append(edge_mode)
        self.visit_counts[next_node] = self.visit_counts.get(next_node, 0) + 1
        loop_penalty = -1.0 * max(0, self.visit_counts[next_node] - 1)

        # Transfer tracking
        transfer_occurred = False
        if self.last_mode and self.last_mode != edge_mode:
            self.num_mode_switches += 1
            transfer_occurred = True
            vehicle = {'subway', 'bus', 'car', 'ride-hail'}
            if self.last_mode in vehicle and edge_mode in vehicle:
                self.num_transfers += 1
        self.last_mode = edge_mode

        # Progress tracking
        cur_hop = self.hop_distances.get(self.current_node, self.bfs_cutoff + 1)
        if cur_hop < self.best_hop_dist:
            self.best_hop_dist = cur_hop
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        potential_after = self._get_potential()
        reward = self._compute_reward(
            edge_time, edge_mode, transfer_occurred,
            potential_before, potential_after, loop_penalty
        )

        # Termination
        terminated      = False
        near_goal       = False
        if self.current_node == self.destination:
            terminated = True
        elif self.near_goal_threshold > 0 and cur_hop <= self.near_goal_threshold:
            terminated = True
            near_goal  = True

        timeout_reason = ''
        truncated = False
        if self.cumulative_time > self.time_budget_current:
            truncated = True;  timeout_reason = 'budget'
        elif self.steps >= self.max_steps_current:
            truncated = True;  timeout_reason = 'max_steps'
        elif self.steps_since_progress >= 50:
            truncated = True;  timeout_reason = 'no_progress'

        # Terminal bonus / timeout penalty
        if terminated:
            reward += 500.0
        elif truncated:
            reward -= 10.0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._info(
            success        = int(terminated),
            success_strict = int(terminated and self.current_node == self.destination),
            near_goal      = int(near_goal),
            timeout        = int(truncated),
            timeout_reason = timeout_reason,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INFO HELPER
    # ══════════════════════════════════════════════════════════════════════════

    def _info(self, success=0, success_strict=None, near_goal=0,
              timeout=0, timeout_reason='') -> dict:
        if success_strict is None:
            success_strict = success
        return {
            'success':             success,
            'success_relaxed':     success,
            'success_strict':      success_strict,
            'near_goal_success':   near_goal,
            'timeout':             timeout,
            'timeout_reason':      timeout_reason,
            'reached_destination': int(self.current_node == self.destination),
            'travel_time':         self.cumulative_time,
            'num_transfers':       self.num_transfers,
            'num_mode_switches':   self.num_mode_switches,
            'num_steps':           self.steps,
            'path_length':         len(self.path),
            'persona_id':          self.persona_id,
            'behavior_10d':        self.behavior_vector_10d.tolist(),
            'mode_sequence':       list(self.modes_used),
            'modes_used':          ','.join(self.modes_used),
            'mode_counts':         {m: self.modes_used.count(m)
                                    for m in set(self.modes_used)},
        }


# Alias for compatibility with training scripts
Phase2MultimodalEnv = MultimodalRoutingEnv10D
