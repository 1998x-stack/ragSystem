#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular RAG System Package
==========================

A modular, industrial-grade Retrieval-Augmented Generation system.

Author: Claude AI
Date: 2025-10-29
"""

__version__ = "1.0.0"
__author__ = "Claude AI"
__description__ = "Modular RAG System with Qwen Models"

# Core imports
from .config import RAGConfig, LoggerManager, load_config
from .rag_pipeline import RAGPipeline

# Component imports
from .dataset_processor import DatasetProcessor
from .embedding_processor import EmbeddingProcessor
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .query_processor import QueryProcessor
from .context_organizer import ContextOrganizer

# Main exports
__all__ = [
    # Core classes
    'RAGConfig',
    'RAGPipeline',
    'LoggerManager',
    'load_config',
    
    # Component classes
    'DatasetProcessor',
    'EmbeddingProcessor',
    'VectorRetriever',
    'BM25Retriever',
    'HybridRetriever',
    'QueryProcessor',
    'ContextOrganizer',
]

# Package metadata
__package_info__ = {
    'name': 'modular_rag',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'models': {
        'llm': 'Qwen/Qwen3-0.6B',
        'embedding': 'Qwen/Qwen3-Embedding-0.6B',
        'dataset': 'rahular/simple-wikipedia'
    },
    'features': [
        'Hybrid retrieval (Vector + BM25)',
        'Query expansion with LLM',
        'Modular architecture',
        'Industrial-grade logging',
        'GPU acceleration support',
        'Comprehensive caching',
        'Performance monitoring'
    ]
}

def get_package_info():
    """Get package information.
    
    Returns:
        Package information dictionary
    """
    return __package_info__.copy()

def quick_start():
    """Quick start guide for the RAG system.
    
    Returns:
        Quick start instructions
    """
    return """
Quick Start Guide:

1. Install dependencies:
   pip install -r requirements.txt

2. Basic usage:
   from modular_rag import RAGConfig, RAGPipeline
   
   config = RAGConfig()
   rag = RAGPipeline(config)
   rag.initialize()
   
   result = rag.query("What is artificial intelligence?")
   print(result['answer'])

3. Custom configuration:
   config = RAGConfig(
       embedding_dim=512,
       device="cuda",
       top_k_retrieval=10
   )

4. From config file:
   config = load_config("config.yaml")

For more examples, see example_modular.py
    """

# Version check
def check_dependencies():
    """Check if required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    deps = {}
    
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = "NOT INSTALLED"
    
    try:
        import transformers
        deps['transformers'] = transformers.__version__
    except ImportError:
        deps['transformers'] = "NOT INSTALLED"
    
    try:
        import faiss
        deps['faiss'] = "AVAILABLE"
    except ImportError:
        deps['faiss'] = "NOT INSTALLED"
    
    try:
        import datasets
        deps['datasets'] = datasets.__version__
    except ImportError:
        deps['datasets'] = "NOT INSTALLED"
    
    try:
        import loguru
        deps['loguru'] = loguru.__version__
    except ImportError:
        deps['loguru'] = "NOT INSTALLED"
    
    try:
        from rank_bm25 import BM25Okapi
        deps['rank_bm25'] = "AVAILABLE"
    except ImportError:
        deps['rank_bm25'] = "NOT INSTALLED"
    
    return deps

# Initialize logging when package is imported
try:
    from .config import LoggerManager
    # Set up basic logging configuration
    import logging
    logging.basicConfig(level=logging.INFO)
except ImportError:
    pass