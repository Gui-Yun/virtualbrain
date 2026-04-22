"""Basal ganglia node grouping metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasalGangliaNodeGroups:
    """Canonical node index groups mirrored from legacy BG setup."""

    igpe_nodes: tuple[int, ...] = (0, 1)
    igpi_nodes: tuple[int, ...] = (2, 3)
    estn_nodes: tuple[int, ...] = (4, 5)
    istr_nodes: tuple[int, ...] = (6, 7)
    eth_nodes: tuple[int, ...] = (8, 9)

    @property
    def inhibitory_nodes(self) -> tuple[int, ...]:
        return self.igpe_nodes + self.igpi_nodes

    @property
    def excitatory_nodes(self) -> tuple[int, ...]:
        return self.estn_nodes + self.eth_nodes

