#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for RAG System
===================================

RAG系统的配置模块，包含所有系统配置参数和日志设置。

Author: Claude AI
Date: 2025-10-29
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from loguru import logger


@dataclass
class RAGConfig:
    """RAG 系统配置类。
    
    包含所有系统运行所需的配置参数，支持从 YAML 文件加载和环境变量覆盖。
    
    Attributes:
        # 模型配置
        dataset_name: 数据集名称
        llm_model_name: 语言模型名称
        embedding_model_name: 嵌入模型名称
        
        # 嵌入配置
        embedding_dim: 嵌入维度 (32-1024)
        batch_size: 批处理大小
        
        # 检索配置
        top_k_retrieval: 检索的文档数量
        bm25_weight: BM25权重 (0-1)
        embedding_weight: 嵌入检索权重 (0-1)
        
        # 文本处理配置
        chunk_size: 文档分块大小
        chunk_overlap: 分块重叠大小
        max_context_length: 最大上下文长度
        min_chunk_length: 最小文档块长度
        
        # 系统配置
        device: 计算设备
        cache_dir: 缓存目录
        log_level: 日志级别
        log_file: 日志文件路径
        
        # 生成配置
        max_new_tokens: 最大生成令牌数
        temperature: 生成温度
        do_sample: 是否使用采样
        
        # 性能配置
        enable_cache: 是否启用缓存
        use_multiprocessing: 是否使用多进程
        num_processes: 进程数量
        max_memory_gb: 最大内存使用(GB)
        
        # BM25参数
        bm25_k1: BM25 k1参数
        bm25_b: BM25 b参数
        
        # 高级配置
        force_rebuild: 是否强制重建索引
        normalize_vectors: 是否标准化向量
        save_intermediate: 是否保存中间结果
    """
    
    # 模型配置
    dataset_name: str = "rahular/simple-wikipedia"
    llm_model_name: str = "Qwen/Qwen3-0.6B"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    
    # 嵌入配置
    embedding_dim: int = 1024
    batch_size: int = 32
    
    # 检索配置
    top_k_retrieval: int = 10
    bm25_weight: float = 0.3
    embedding_weight: float = 0.7
    
    # 文本处理配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_length: int = 4096
    min_chunk_length: int = 50
    
    # 系统配置
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    log_file: str = "./logs/rag_system.log"
    
    # 生成配置
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    
    # 性能配置
    enable_cache: bool = True
    use_multiprocessing: bool = False
    num_processes: int = 4
    max_memory_gb: float = 8.0
    
    # BM25参数
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    
    # 高级配置
    force_rebuild: bool = False
    normalize_vectors: bool = True
    save_intermediate: bool = True
    
    def __post_init__(self) -> None:
        """初始化后处理，验证配置参数并创建必要目录。"""
        self._validate_config()
        self._create_directories()
        self._normalize_weights()
    
    def _validate_config(self) -> None:
        """验证配置参数的有效性。
        
        Raises:
            ValueError: 当配置参数无效时抛出
        """
        # 验证嵌入维度
        if not (32 <= self.embedding_dim <= 1024):
            raise ValueError(f"embedding_dim 必须在 32-1024 范围内，当前值: {self.embedding_dim}")
        
        # 验证权重
        if not (0.0 <= self.bm25_weight <= 1.0):
            raise ValueError(f"bm25_weight 必须在 0-1 范围内，当前值: {self.bm25_weight}")
        
        if not (0.0 <= self.embedding_weight <= 1.0):
            raise ValueError(f"embedding_weight 必须在 0-1 范围内，当前值: {self.embedding_weight}")
        
        # 验证检索参数
        if self.top_k_retrieval <= 0:
            raise ValueError(f"top_k_retrieval 必须大于 0，当前值: {self.top_k_retrieval}")
        
        # 验证文本处理参数
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size 必须大于 0，当前值: {self.chunk_size}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) 不能大于等于 chunk_size ({self.chunk_size})")
        
        # 验证设备
        if self.device not in ["cpu", "cuda", "auto"]:
            logger.warning(f"设备 '{self.device}' 可能不被支持，建议使用 'cpu' 或 'cuda'")
        
        # 验证温度参数
        if not (0.0 <= self.temperature <= 2.0):
            logger.warning(f"temperature ({self.temperature}) 超出建议范围 [0.0, 2.0]")
    
    def _create_directories(self) -> None:
        """创建必要的目录。"""
        directories = [
            self.cache_dir,
            os.path.dirname(self.log_file) if self.log_file else "./logs"
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _normalize_weights(self) -> None:
        """标准化权重参数，确保总和为1。"""
        total_weight = self.bm25_weight + self.embedding_weight
        if total_weight > 0:
            self.bm25_weight = self.bm25_weight / total_weight
            self.embedding_weight = self.embedding_weight / total_weight
        else:
            # 默认权重
            self.bm25_weight = 0.3
            self.embedding_weight = 0.7
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGConfig":
        """从 YAML 文件加载配置。
        
        Args:
            config_path: YAML配置文件路径
            
        Returns:
            配置对象实例
            
        Raises:
            FileNotFoundError: 当配置文件不存在时抛出
            ValueError: 当配置格式错误时抛出
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 展平嵌套的配置结构
            flattened_config = cls._flatten_config(config_data)
            
            # 从环境变量覆盖配置
            env_config = cls._load_from_env()
            flattened_config.update(env_config)
            
            # 创建配置实例
            return cls(**flattened_config)
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 配置格式错误: {e}")
        except Exception as e:
            raise ValueError(f"配置加载失败: {e}")
    
    @staticmethod
    def _flatten_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """展平嵌套的配置字典。
        
        Args:
            config_data: 嵌套的配置字典
            
        Returns:
            展平后的配置字典
        """
        flattened = {}
        
        # 处理嵌套的配置结构
        sections = {
            'models': ['dataset_name', 'llm_model_name', 'embedding_model_name'],
            'embedding': ['embedding_dim', 'batch_size'],
            'retrieval': ['top_k_retrieval', 'bm25_weight', 'embedding_weight'],
            'text_processing': ['chunk_size', 'chunk_overlap', 'max_context_length', 'min_chunk_length'],
            'system': ['device', 'cache_dir', 'log_level', 'log_file'],
            'generation': ['max_new_tokens', 'temperature', 'do_sample'],
            'performance': ['enable_cache', 'use_multiprocessing', 'num_processes', 'max_memory_gb'],
            'bm25': ['bm25_k1', 'bm25_b'],
            'advanced': ['force_rebuild', 'normalize_vectors', 'save_intermediate']
        }
        
        for section, keys in sections.items():
            if section in config_data:
                for key in keys:
                    if key in config_data[section]:
                        flattened[key] = config_data[section][key]
        
        # 处理直接的键值对
        for key, value in config_data.items():
            if not isinstance(value, dict):
                flattened[key] = value
        
        return flattened
    
    @staticmethod
    def _load_from_env() -> Dict[str, Any]:
        """从环境变量加载配置。
        
        环境变量格式: RAG_<CONFIG_KEY>=<VALUE>
        
        Returns:
            从环境变量提取的配置字典
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith("RAG_"):
                config_key = key[4:].lower()  # 移除 RAG_ 前缀并转换为小写
                
                # 尝试转换类型
                if value.lower() in ['true', 'false']:
                    env_config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    env_config[config_key] = int(value)
                elif value.replace('.', '').isdigit():
                    env_config[config_key] = float(value)
                else:
                    env_config[config_key] = value
        
        return env_config
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典。
        
        Returns:
            配置字典
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_yaml(self, config_path: str) -> None:
        """将配置保存到 YAML 文件。
        
        Args:
            config_path: 保存路径
        """
        config_data = {
            'models': {
                'dataset_name': self.dataset_name,
                'llm_model_name': self.llm_model_name,
                'embedding_model_name': self.embedding_model_name
            },
            'embedding': {
                'embedding_dim': self.embedding_dim,
                'batch_size': self.batch_size
            },
            'retrieval': {
                'top_k_retrieval': self.top_k_retrieval,
                'bm25_weight': self.bm25_weight,
                'embedding_weight': self.embedding_weight
            },
            'text_processing': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'max_context_length': self.max_context_length,
                'min_chunk_length': self.min_chunk_length
            },
            'system': {
                'device': self.device,
                'cache_dir': self.cache_dir,
                'log_level': self.log_level,
                'log_file': self.log_file
            },
            'generation': {
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'do_sample': self.do_sample
            },
            'performance': {
                'enable_cache': self.enable_cache,
                'use_multiprocessing': self.use_multiprocessing,
                'num_processes': self.num_processes,
                'max_memory_gb': self.max_memory_gb
            },
            'bm25': {
                'bm25_k1': self.bm25_k1,
                'bm25_b': self.bm25_b
            },
            'advanced': {
                'force_rebuild': self.force_rebuild,
                'normalize_vectors': self.normalize_vectors,
                'save_intermediate': self.save_intermediate
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)


class LoggerManager:
    """日志管理器，统一管理系统日志配置。"""
    
    _initialized = False
    
    @classmethod
    def setup_logger(cls, config: RAGConfig) -> None:
        """设置 loguru 日志系统。
        
        Args:
            config: RAG配置对象
        """
        if cls._initialized:
            return
        
        # 移除默认的控制台日志处理器
        logger.remove()
        
        # 设置日志级别
        log_level = config.log_level.upper()
        
        # 添加控制台日志处理器
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # 添加文件日志处理器
        if config.log_file:
            logger.add(
                config.log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                encoding="utf-8"
            )
        
        # 记录初始化信息
        logger.info("日志系统初始化完成")
        logger.info(f"日志级别: {log_level}")
        logger.info(f"日志文件: {config.log_file}")
        logger.info(f"设备: {config.device}")
        logger.info(f"缓存目录: {config.cache_dir}")
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logger:
        """获取指定名称的日志记录器。
        
        Args:
            name: 日志记录器名称
            
        Returns:
            配置好的日志记录器
        """
        return logger.bind(name=name)


def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """加载系统配置。
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        RAG配置对象
    """
    if config_path and os.path.exists(config_path):
        config = RAGConfig.from_yaml(config_path)
        logger.info(f"从文件加载配置: {config_path}")
    else:
        config = RAGConfig()
        logger.info("使用默认配置")
        
        if config_path:
            logger.warning(f"配置文件不存在: {config_path}")
    
    # 初始化日志系统
    LoggerManager.setup_logger(config)
    
    return config


# 导出主要类和函数
__all__ = [
    'RAGConfig',
    'LoggerManager',
    'load_config'
]