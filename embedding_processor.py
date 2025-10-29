#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Processor Module for RAG System
=========================================

嵌入处理器模块，负责生成和管理文档嵌入向量。

Author: Claude AI
Date: 2025-10-29
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from config import RAGConfig


class EmbeddingProcessor:
    """嵌入处理器，负责生成和管理文档嵌入向量。
    
    主要功能：
    1. 加载和管理嵌入模型
    2. 批量文本编码
    3. 嵌入向量管理和缓存
    4. 性能优化和内存管理
    
    Attributes:
        config: RAG系统配置对象
        model: 嵌入模型
        tokenizer: 分词器
        embeddings: 生成的嵌入矩阵
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化嵌入处理器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.embeddings: Optional[np.ndarray] = None
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 设备配置
        self.device = torch.device(self.config.device)
        
        self._logger.info(f"嵌入处理器初始化完成，使用设备: {self.device}")
    
    def load_model(self) -> None:
        """加载嵌入模型和分词器。
        
        Raises:
            RuntimeError: 当模型加载失败时抛出
        """
        try:
            self._logger.info(f"正在加载嵌入模型: {self.config.embedding_model_name}")
            start_time = time.time()
            
            # 加载分词器
            self._logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            # 加载模型
            self._logger.info("加载嵌入模型...")
            self.model = AutoModel.from_pretrained(
                self.config.embedding_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # 移动模型到指定设备
            if self.device.type != "cuda" or not hasattr(self.model, 'device_map'):
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            load_time = time.time() - start_time
            
            # 显示模型信息
            self._log_model_info()
            
            self._logger.info(f"嵌入模型加载完成，耗时: {load_time:.2f} 秒")
            
            # 测试模型
            self._test_model()
            
        except Exception as e:
            self._logger.error(f"嵌入模型加载失败: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _log_model_info(self) -> None:
        """记录模型信息。"""
        if self.model is None:
            return
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self._logger.info(f"模型参数统计:")
        self._logger.info(f"  - 总参数数: {total_params:,}")
        self._logger.info(f"  - 可训练参数: {trainable_params:,}")
        self._logger.info(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        # 显示模型配置
        if hasattr(self.model, 'config'):
            config = self.model.config
            self._logger.info(f"模型配置:")
            if hasattr(config, 'hidden_size'):
                self._logger.info(f"  - 隐藏层大小: {config.hidden_size}")
            if hasattr(config, 'num_hidden_layers'):
                self._logger.info(f"  - 隐藏层数: {config.num_hidden_layers}")
            if hasattr(config, 'num_attention_heads'):
                self._logger.info(f"  - 注意力头数: {config.num_attention_heads}")
    
    def _test_model(self) -> None:
        """测试模型是否正常工作。"""
        try:
            test_text = "这是一个测试文本。"
            test_embedding = self.encode_texts([test_text])
            
            self._logger.info(f"模型测试通过，嵌入维度: {test_embedding.shape}")
            
        except Exception as e:
            self._logger.error(f"模型测试失败: {e}")
            raise RuntimeError(f"Model test failed: {e}")
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """对文本列表进行编码。
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，如果为None则使用配置中的值
            show_progress: 是否显示进度
            
        Returns:
            文本嵌入矩阵，形状为 (n_texts, embedding_dim)
            
        Raises:
            ValueError: 当模型未加载时抛出
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        if not texts:
            return np.array([]).reshape(0, self.config.embedding_dim)
        
        batch_size = batch_size or self.config.batch_size
        
        self._logger.info(f"开始编码 {len(texts):,} 个文本，批大小: {batch_size}")
        start_time = time.time()
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if show_progress and batch_num % max(1, total_batches // 10) == 0:
                    self._logger.info(f"编码进度: {batch_num}/{total_batches} ({batch_num/total_batches*100:.1f}%)")
                
                try:
                    # 编码当前批次
                    batch_embeddings = self._encode_batch(batch_texts)
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    self._logger.error(f"编码批次 {batch_num} 失败: {e}")
                    # 使用零向量作为fallback
                    fallback_embeddings = np.zeros((len(batch_texts), self.config.embedding_dim))
                    embeddings.append(fallback_embeddings)
                    
                # 内存管理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合并所有批次的嵌入
        embeddings_matrix = np.vstack(embeddings) if embeddings else np.array([])
        encoding_time = time.time() - start_time
        
        self._logger.info(f"文本编码完成:")
        self._logger.info(f"  - 嵌入维度: {embeddings_matrix.shape}")
        self._logger.info(f"  - 编码时间: {encoding_time:.2f} 秒")
        self._logger.info(f"  - 平均速度: {len(texts)/encoding_time:.1f} 文本/秒")
        
        return embeddings_matrix
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """编码单个批次的文本。
        
        Args:
            texts: 文本列表
            
        Returns:
            批次嵌入矩阵
        """
        # 预处理文本
        texts = [self._preprocess_text(text) for text in texts]
        
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Qwen embedding model 的推荐长度
            return_tensors="pt"
        ).to(self.device)
        
        # 获取模型输出
        outputs = self.model(**inputs)
        
        # 提取嵌入向量
        embeddings = self._extract_embeddings(outputs, inputs['attention_mask'])
        
        # 标准化嵌入（如果配置要求）
        if self.config.normalize_vectors:
            embeddings = self._normalize_embeddings(embeddings)
        
        # 调整嵌入维度（如果需要）
        if embeddings.shape[1] != self.config.embedding_dim:
            embeddings = self._adjust_embedding_dimension(embeddings)
        
        return embeddings.cpu().numpy()
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本。
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 基本清理
        text = text.strip()
        
        # 移除过多的空白字符
        text = ' '.join(text.split())
        
        # 限制长度（分词器会截断，但提前限制可以提高效率）
        max_chars = 512 * 4  # 估算：每个token约4个字符
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text
    
    def _extract_embeddings(
        self, 
        outputs: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """从模型输出中提取嵌入向量。
        
        Args:
            outputs: 模型输出
            attention_mask: 注意力掩码
            
        Returns:
            提取的嵌入向量
        """
        # 获取最后一层的隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            # 如果有 pooler_output，直接使用
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            else:
                raise ValueError("无法从模型输出中提取嵌入向量")
        
        # 使用注意力掩码进行平均池化
        embeddings = self._mean_pooling(hidden_states, attention_mask)
        
        return embeddings
    
    def _mean_pooling(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """执行平均池化。
        
        Args:
            hidden_states: 隐藏状态张量
            attention_mask: 注意力掩码
            
        Returns:
            池化后的嵌入向量
        """
        # 扩展注意力掩码以匹配隐藏状态的维度
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # 计算加权平均
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """标准化嵌入向量。
        
        Args:
            embeddings: 原始嵌入向量
            
        Returns:
            标准化后的嵌入向量
        """
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def _adjust_embedding_dimension(self, embeddings: torch.Tensor) -> torch.Tensor:
        """调整嵌入维度以匹配配置。
        
        Args:
            embeddings: 原始嵌入向量
            
        Returns:
            调整后的嵌入向量
        """
        current_dim = embeddings.shape[1]
        target_dim = self.config.embedding_dim
        
        if current_dim == target_dim:
            return embeddings
        
        if current_dim > target_dim:
            # 截断到目标维度
            self._logger.warning(f"截断嵌入维度从 {current_dim} 到 {target_dim}")
            return embeddings[:, :target_dim]
        else:
            # 填充到目标维度
            self._logger.warning(f"填充嵌入维度从 {current_dim} 到 {target_dim}")
            padding = torch.zeros(embeddings.shape[0], target_dim - current_dim, device=embeddings.device)
            return torch.cat([embeddings, padding], dim=1)
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str = "embeddings.npy") -> None:
        """保存嵌入到文件。
        
        Args:
            embeddings: 嵌入矩阵
            filename: 文件名
        """
        filepath = os.path.join(self.config.cache_dir, filename)
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存嵌入
            np.save(filepath, embeddings)
            
            # 保存元数据
            metadata = {
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'config': {
                    'embedding_dim': self.config.embedding_dim,
                    'embedding_model_name': self.config.embedding_model_name,
                    'normalize_vectors': self.config.normalize_vectors
                },
                'timestamp': time.time()
            }
            
            metadata_path = filepath.replace('.npy', '_metadata.npy')
            np.save(metadata_path, metadata, allow_pickle=True)
            
            self._logger.info(f"嵌入已保存到: {filepath}")
            self._logger.info(f"嵌入形状: {embeddings.shape}")
            
        except Exception as e:
            self._logger.error(f"保存嵌入失败: {e}")
    
    def load_embeddings(self, filename: str = "embeddings.npy") -> Optional[np.ndarray]:
        """从文件加载嵌入。
        
        Args:
            filename: 文件名
            
        Returns:
            加载的嵌入矩阵，如果失败则返回 None
        """
        filepath = os.path.join(self.config.cache_dir, filename)
        metadata_path = filepath.replace('.npy', '_metadata.npy')
        
        if not os.path.exists(filepath):
            self._logger.info(f"嵌入文件不存在: {filepath}")
            return None
        
        try:
            # 检查元数据兼容性
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                if not self._is_embedding_compatible(metadata):
                    self._logger.warning("嵌入配置不兼容，需要重新生成")
                    return None
            
            # 加载嵌入
            embeddings = np.load(filepath)
            
            self._logger.info(f"从 {filepath} 加载了嵌入")
            self._logger.info(f"嵌入形状: {embeddings.shape}")
            
            # 验证嵌入维度
            if embeddings.shape[1] != self.config.embedding_dim:
                self._logger.warning(f"嵌入维度不匹配: {embeddings.shape[1]} vs {self.config.embedding_dim}")
                return None
            
            return embeddings
            
        except Exception as e:
            self._logger.error(f"加载嵌入失败: {e}")
            return None
    
    def _is_embedding_compatible(self, metadata: dict) -> bool:
        """检查嵌入是否与当前配置兼容。
        
        Args:
            metadata: 嵌入元数据
            
        Returns:
            是否兼容
        """
        config = metadata.get('config', {})
        
        # 检查关键配置
        if config.get('embedding_dim') != self.config.embedding_dim:
            return False
        
        if config.get('embedding_model_name') != self.config.embedding_model_name:
            return False
        
        if config.get('normalize_vectors') != self.config.normalize_vectors:
            return False
        
        return True
    
    def get_memory_usage(self) -> dict:
        """获取内存使用情况。
        
        Returns:
            内存使用统计
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        if self.embeddings is not None:
            memory_info['embeddings_size'] = self.embeddings.nbytes / 1024**3  # GB
        
        return memory_info
    
    def clear_cache(self) -> None:
        """清理内存和缓存。"""
        self.embeddings = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._logger.info("缓存已清理")


# 导出主要类
__all__ = ['EmbeddingProcessor']