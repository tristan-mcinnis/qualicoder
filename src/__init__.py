"""
Qualitative Coding System

A comprehensive system for qualitative analysis of text data using:
- OpenAI API for code generation
- Pinecone for vector storage
- HuggingFace transformers for embeddings
- Sentence-boundary preserving chunking
"""

from .qualitative_coder import QualitativeCoder
from .config import Config
from .logger import logger
from .exporters import ResultExporter
from .transcript_processor import TranscriptProcessor

__version__ = "1.0.0"
__author__ = "Qualitative Research Team"

__all__ = [
    'QualitativeCoder',
    'Config', 
    'logger',
    'ResultExporter',
    'TranscriptProcessor'
]