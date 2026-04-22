"""Next-generation multiscale co-simulation package.

This package is a clean-room implementation that keeps the key co-simulation
logic while modernizing architecture and interfaces.
"""

from .core.orchestrator import CoSimulationOrchestrator

__all__ = ["CoSimulationOrchestrator"]

