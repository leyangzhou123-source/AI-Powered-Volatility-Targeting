"""Base engine class."""

from abc import ABC, abstractmethod


class Engine(ABC):
    """Abstract base class for backtest engines."""
    
    @abstractmethod
    def run(self, returns):
        """Run the backtest and return a DataFrame."""
        pass
    
    @abstractmethod
    def save(self):
        """Save results to disk."""
        pass
    
    @abstractmethod
    def summary(self):
        """Print a summary of the results."""
        pass
