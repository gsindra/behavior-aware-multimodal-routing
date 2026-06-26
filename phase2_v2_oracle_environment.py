"""
================================================================================
PHASE 2 — V2 ORACLE ENVIRONMENT
================================================================================
Wraps MultimodalRoutingEnv10D and injects the oracle-recommended action
at position 0 of the candidate set at every step.

Oracle mode:
    inject  — oracle action inserted at position 0, policy can still choose
              any candidate (same as training configuration)
    strict  — policy is forced to take oracle action (evaluation only)

The oracle table is a dict: {dest_node: {src_node: next_step_node}}
Loaded from: Results/Phase 2 Rerun Clean/Phase2_Oracle/phase2_oracle_500dest.pickle

Design:
    - Inherits all 8 fixes from MultimodalRoutingEnv10D (no P2 hacking,
      BFS shaping, bounded potential, street-node sampling etc.)
    - Only change: _get_candidates() injects oracle action at index 0
    - If oracle not found for current state, falls back to base ranking

Author: Indramuthu Sundaram — Phase 2, North Carolina A&T State University
================================================================================
"""

import numpy as np
import networkx as nx
import logging
from typing import Dict, Optional

from phase2_environment_10d_clean import MultimodalRoutingEnv10D

logger = logging.getLogger(__name__)


class MultimodalRoutingEnvV2Oracle(MultimodalRoutingEnv10D):
    """
    Oracle-guided routing environment.

    Extends the clean Phase 2 environment by injecting the oracle-recommended
    next hop at position 0 of the candidate set. The policy still selects
    freely from all candidates — it just always has the oracle action available.

    Args:
        oracle_table : dict  {dest_node: {src_node: next_step_node}}
        oracle_mode  : str   'inject' (default) or 'strict'
        **kwargs     : passed directly to MultimodalRoutingEnv10D
    """

    def __init__(
        self,
        oracle_table: Dict,
        oracle_mode:  str = 'inject',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.oracle_table = oracle_table
        self.oracle_mode  = oracle_mode
        self._oracle_hits  = 0
        self._oracle_misses = 0
        logger.info(f"V2Oracle env: mode={oracle_mode}  "
                    f"oracle_dests={len(oracle_table)}")

    def _get_candidates(self, node):
        """
        Get behavior-ranked candidates, then inject oracle action at position 0.

        If oracle has a recommendation for (destination, current_node):
            - Remove it from wherever it sits in the ranked list
            - Insert it at position 0
        If oracle has no recommendation: fall back to base ranking unchanged.
        """
        # Get base behavior-ranked candidates from parent class
        candidates = super()._get_candidates(node)

        # Look up oracle recommendation
        oracle_next = self._get_oracle_action(node)

        if oracle_next is not None and oracle_next != node:
            self._oracle_hits += 1
            # Remove oracle node from its current position if present
            candidates = [c for c in candidates if c != oracle_next]
            # Insert at position 0
            candidates = [oracle_next] + candidates
            # Trim to top_k
            candidates = candidates[:self.top_k]
        else:
            self._oracle_misses += 1

        return candidates

    def _get_oracle_action(self, current_node) -> Optional:
        """
        Look up oracle next step for (destination, current_node).

        Oracle table structure:
            {dest_node: {src_node: next_step_node}}
        """
        if not self.oracle_table:
            return None

        dest = getattr(self, 'destination', None)
        if dest is None:
            return None

        dest_table = self.oracle_table.get(dest)
        if dest_table is None:
            return None

        return dest_table.get(current_node)

    def step(self, action: int):
        """
        In strict mode: override action with oracle recommendation.
        In inject mode: use action as-is (oracle is just at position 0).
        """
        if self.oracle_mode == 'strict':
            oracle_next = self._get_oracle_action(self.current_node)
            if oracle_next is not None and self._candidates:
                # Find oracle position in candidates
                try:
                    action = self._candidates.index(oracle_next)
                except ValueError:
                    pass  # oracle not in candidates — use policy action

        return super().step(action)

    def reset(self, seed=None, options=None):
        """Reset episode state including oracle hit counters."""
        self._oracle_hits   = 0
        self._oracle_misses = 0
        return super().reset(seed=seed, options=options)

    def get_oracle_coverage(self) -> float:
        """Return oracle coverage rate for the current episode."""
        total = self._oracle_hits + self._oracle_misses
        if total == 0:
            return 0.0
        return self._oracle_hits / total


# Alias for backward compatibility
Phase2OracleEnv = MultimodalRoutingEnvV2Oracle
