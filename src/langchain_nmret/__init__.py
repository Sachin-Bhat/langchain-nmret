"""
An unofficial LangChain based Retriever built using concepts from the Titans Neural Memory
"""

__version__ = "0.1.1"

from .nmret import (
    LightThinkerCompressor,
    NeuralMemoryRetriever,
    TitansNeuralMemoryWrapper,
    VectorStoreContextualMemory,
)

__all__ = [
    "TitansNeuralMemoryWrapper",
    "VectorStoreContextualMemory",
    "LightThinkerCompressor",
    "NeuralMemoryRetriever",
]
