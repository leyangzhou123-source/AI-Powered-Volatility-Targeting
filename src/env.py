"""Central environment configuration."""

from pathlib import Path


ROOT = Path(__file__).parents[1]

PATHS = {
    "root": ROOT,
    "raw": ROOT / "data" / "raw",
    "processed": ROOT / "data" / "processed",
    "prices": ROOT / "data" / "raw" / "prices",
    "features": ROOT / "data" / "processed" / "features",
    "cache": ROOT / "data" / "cache",
    "results": ROOT / "results",
    "evaluation": ROOT / "results" / "evaluation",
    "router_logs": ROOT / "results" / "evaluation" / "router_logs",
    "strategies": ROOT / "configs" / "strategies",
}


class Env:
    """Central environment manager."""
    
    @classmethod
    def path(cls, key, source=None):
        """Fetch path by key, optionally with source subdirectory."""
        p = PATHS[key]
        return p / source if source else p
