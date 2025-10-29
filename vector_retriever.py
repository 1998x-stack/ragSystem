#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Retriever Module for RAG System
======================================

向量检索器模块，使用 FAISS 进行高效的向量相似度搜索。

Author: Claude AI
Date: 2025-10-29
"""

import os
import time
from typing import Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from config import RAGConfig


class VectorRetriever:
    """向量检索器，使用 FAISS 进行相似度搜索。
    
    主要功能：
    1. 构建和管理 FAISS 索引
    2. 高效向量相似度搜索
    3. 索引持久化和加载
    4. 性能优化和内存管理
    
    Attributes:
        config: RAG系统配置对象
        index: FAISS索引对象
        embeddings: 原始嵌入矩阵
        dimension: 向量维度
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化向量检索器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.dimension = config.embedding_dim
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 检查 FAISS 可用性
        self._check_faiss_installation()
        
        self._logger.info(f"向量检索器初始化完成，维度: {self.dimension}")
    
    def _check_faiss_installation(self) -> None:
        """检查 FAISS 安装和 GPU 支持。"""
        try:
            # 检查基本 FAISS 功能
            test_dim = 128
            test_index = faiss.IndexFlatL2(test_dim)
            
            # 检查 GPU 支持
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                self._logger.info(f"FAISS GPU 支持可用，GPU 数量: {faiss.get_num_gpus()}")
            else:
                self._logger.info("FAISS 使用 CPU 模式")
                
        except Exception as e:
            self._logger.error(f"FAISS 检查失败: {e}")
            raise RuntimeError(f"FAISS installation check failed: {e}")
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "flat") -> None:
        """构建 FAISS 索引。
        
        Args:
            embeddings: 文档嵌入矩阵，形状为 (n_docs, embedding_dim)
            index_type: 索引类型 ("flat", "ivf", "hnsw")
            
        Raises:
            ValueError: 当嵌入维度不匹配时抛出
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"嵌入维度不匹配: {embeddings.shape[1]} vs {self.dimension}")
        
        self._logger.info(f"正在构建 {index_type.upper()} 索引...")
        start_time = time.time()
        
        # 预处理嵌入
        processed_embeddings = self._preprocess_embeddings(embeddings)
        
        # 创建索引
        self.index = self._create_index(index_type, processed_embeddings.shape[0])
        
        # 添加向量到索引
        self._add_vectors_to_index(processed_embeddings)
        
        # 保存嵌入副本
        self.embeddings = processed_embeddings.copy()
        
        build_time = time.time() - start_time
        
        self._logger.info(f"FAISS 索引构建完成:")
        self._logger.info(f"  - 索引类型: {index_type.upper()}")
        self._logger.info(f"  - 向量数量: {self.index.ntotal:,}")
        self._logger.info(f"  - 向量维度: {self.dimension}")
        self._logger.info(f"  - 构建时间: {build_time:.2f} 秒")
        self._logger.info(f"  - 索引大小: {self._get_index_memory_usage():.1f} MB")
    
    def _preprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """预处理嵌入向量。
        
        Args:
            embeddings: 原始嵌入矩阵
            
        Returns:
            预处理后的嵌入矩阵
        """
        # 转换为 float32（FAISS 要求）
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # 标准化向量（用于余弦相似度）
        if self.config.normalize_vectors:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # 避免除零
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
            self._logger.debug("嵌入向量已标准化")
        
        # 检查向量质量
        self._validate_embeddings(embeddings)
        
        return embeddings
    
    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """验证嵌入向量质量。
        
        Args:
            embeddings: 嵌入矩阵
            
        Raises:
            ValueError: 当嵌入质量不符合要求时抛出
        """
        # 检查 NaN 和无穷值
        if np.any(np.isnan(embeddings)):
            raise ValueError("嵌入中包含 NaN 值")
        
        if np.any(np.isinf(embeddings)):
            raise ValueError("嵌入中包含无穷值")
        
        # 检查零向量
        zero_vectors = np.all(embeddings == 0, axis=1).sum()
        if zero_vectors > 0:
            self._logger.warning(f"发现 {zero_vectors} 个零向量")
        
        # 统计信息
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
        self._logger.debug(f"嵌入统计: 平均模长={mean_norm:.4f}")
    
    def _create_index(self, index_type: str, num_vectors: int) -> faiss.Index:
        """创建 FAISS 索引。
        
        Args:
            index_type: 索引类型
            num_vectors: 向量数量
            
        Returns:
            创建的 FAISS 索引
        """
        if index_type.lower() == "flat":
            # 精确搜索索引
            if self.config.normalize_vectors:
                # 内积索引（标准化向量相当于余弦相似度）
                index = faiss.IndexFlatIP(self.dimension)
            else:
                # L2 距离索引
                index = faiss.IndexFlatL2(self.dimension)
                
        elif index_type.lower() == "ivf":
            # IVF 索引（近似搜索，适合大规模数据）
            nlist = min(max(int(np.sqrt(num_vectors)), 100), 65536)
            
            if self.config.normalize_vectors:
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            self._logger.info(f"IVF 索引参数: nlist={nlist}")
            
        elif index_type.lower() == "hnsw":
            # HNSW 索引（高效近似搜索）
            m = 16  # 连接数
            index = faiss.IndexHNSWFlat(self.dimension, m)
            index.hnsw.efConstruction = 200
            
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # GPU 加速（如果可用）
        if (self.config.device == "cuda" and 
            hasattr(faiss, 'get_num_gpus') and 
            faiss.get_num_gpus() > 0):
            try:
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                self._logger.info("索引已移动到 GPU")
                return gpu_index
            except Exception as e:
                self._logger.warning(f"GPU 索引创建失败，使用 CPU: {e}")
        
        return index
    
    def _add_vectors_to_index(self, embeddings: np.ndarray) -> None:
        """将向量添加到索引。
        
        Args:
            embeddings: 嵌入矩阵
        """
        # 对于 IVF 索引，需要先训练
        if hasattr(self.index, 'train'):
            if not self.index.is_trained:
                self._logger.info("训练 IVF 索引...")
                self.index.train(embeddings)
        
        # 批量添加向量
        batch_size = 10000  # 避免内存问题
        num_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            
            self.index.add(batch_embeddings)
            
            if num_batches > 1:
                self._logger.debug(f"添加批次 {i//batch_size + 1}/{num_batches}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int,
        search_params: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相似的文档。
        
        Args:
            query_embedding: 查询嵌入向量，形状为 (1, embedding_dim) 或 (embedding_dim,)
            k: 返回的文档数量
            search_params: 搜索参数（可选）
            
        Returns:
            (相似度分数, 文档索引)
            
        Raises:
            ValueError: 当索引未构建时抛出
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        # 预处理查询向量
        query_vector = self._preprocess_query(query_embedding)
        
        # 设置搜索参数
        if search_params:
            self._set_search_params(search_params)
        
        # 执行搜索
        start_time = time.time()
        scores, indices = self.index.search(query_vector, k)
        search_time = time.time() - start_time
        
        # 处理搜索结果
        scores = scores[0]  # 移除批次维度
        indices = indices[0]
        
        # 过滤无效结果
        valid_mask = indices >= 0
        scores = scores[valid_mask]
        indices = indices[valid_mask]
        
        self._logger.debug(f"向量搜索完成，找到 {len(indices)} 个结果，耗时 {search_time:.4f} 秒")
        
        return scores, indices
    
    def _preprocess_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """预处理查询向量。
        
        Args:
            query_embedding: 原始查询向量
            
        Returns:
            预处理后的查询向量
        """
        # 确保形状正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 转换数据类型
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # 标准化（如果配置要求）
        if self.config.normalize_vectors:
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        return query_embedding
    
    def _set_search_params(self, search_params: dict) -> None:
        """设置搜索参数。
        
        Args:
            search_params: 搜索参数字典
        """
        # 为不同类型的索引设置参数
        if hasattr(self.index, 'nprobe'):
            # IVF 索引参数
            if 'nprobe' in search_params:
                self.index.nprobe = search_params['nprobe']
        
        if hasattr(self.index, 'hnsw'):
            # HNSW 索引参数
            if 'ef' in search_params:
                self.index.hnsw.efSearch = search_params['ef']
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        k: int,
        batch_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量搜索。
        
        Args:
            query_embeddings: 查询嵌入矩阵，形状为 (n_queries, embedding_dim)
            k: 每个查询返回的文档数量
            batch_size: 批处理大小
            
        Returns:
            (相似度分数矩阵, 文档索引矩阵)
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        n_queries = query_embeddings.shape[0]
        all_scores = []
        all_indices = []
        
        self._logger.info(f"批量搜索 {n_queries} 个查询")
        
        for i in range(0, n_queries, batch_size):
            batch_end = min(i + batch_size, n_queries)
            batch_queries = query_embeddings[i:batch_end]
            
            # 预处理批次查询
            batch_queries = self._preprocess_query(batch_queries)
            
            # 执行搜索
            scores, indices = self.index.search(batch_queries, k)
            
            all_scores.append(scores)
            all_indices.append(indices)
        
        # 合并结果
        all_scores = np.vstack(all_scores)
        all_indices = np.vstack(all_indices)
        
        return all_scores, all_indices
    
    def get_vector_by_id(self, vector_id: int) -> Optional[np.ndarray]:
        """根据ID获取向量。
        
        Args:
            vector_id: 向量ID
            
        Returns:
            向量，如果ID无效则返回None
        """
        if self.embeddings is None or vector_id >= len(self.embeddings):
            return None
        
        return self.embeddings[vector_id]
    
    def _get_index_memory_usage(self) -> float:
        """获取索引内存使用量（MB）。
        
        Returns:
            内存使用量（MB）
        """
        if self.index is None:
            return 0.0
        
        # 估算索引大小
        num_vectors = self.index.ntotal
        vector_size = self.dimension * 4  # float32
        
        # 基础向量存储
        base_memory = num_vectors * vector_size / 1024 / 1024
        
        # 索引结构开销（粗略估算）
        overhead_factor = 1.2  # 20% 开销
        
        return base_memory * overhead_factor
    
    def save_index(self, filename: str = "faiss_index") -> None:
        """保存 FAISS 索引。
        
        Args:
            filename: 文件名前缀
        """
        if self.index is None:
            self._logger.warning("索引未构建，无法保存")
            return
        
        index_path = os.path.join(self.config.cache_dir, f"{filename}.faiss")
        embeddings_path = os.path.join(self.config.cache_dir, f"{filename}_embeddings.npy")
        metadata_path = os.path.join(self.config.cache_dir, f"{filename}_metadata.npy")
        
        try:
            # 确保目录存在
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # 保存索引（如果是 GPU 索引，先移动到 CPU）
            index_to_save = self.index
            if hasattr(self.index, 'index'):  # GPU 索引
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            
            faiss.write_index(index_to_save, index_path)
            
            # 保存嵌入
            if self.embeddings is not None:
                np.save(embeddings_path, self.embeddings)
            
            # 保存元数据
            metadata = {
                'dimension': self.dimension,
                'num_vectors': self.index.ntotal,
                'index_type': type(self.index).__name__,
                'config': {
                    'embedding_dim': self.config.embedding_dim,
                    'normalize_vectors': self.config.normalize_vectors
                },
                'timestamp': time.time()
            }
            np.save(metadata_path, metadata, allow_pickle=True)
            
            self._logger.info(f"FAISS 索引已保存:")
            self._logger.info(f"  - 索引文件: {index_path}")
            self._logger.info(f"  - 嵌入文件: {embeddings_path}")
            self._logger.info(f"  - 元数据文件: {metadata_path}")
            
        except Exception as e:
            self._logger.error(f"保存 FAISS 索引失败: {e}")
    
    def load_index(self, filename: str = "faiss_index") -> bool:
        """加载 FAISS 索引。
        
        Args:
            filename: 文件名前缀
            
        Returns:
            是否成功加载
        """
        index_path = os.path.join(self.config.cache_dir, f"{filename}.faiss")
        embeddings_path = os.path.join(self.config.cache_dir, f"{filename}_embeddings.npy")
        metadata_path = os.path.join(self.config.cache_dir, f"{filename}_metadata.npy")
        
        if not os.path.exists(index_path):
            self._logger.info(f"索引文件不存在: {index_path}")
            return False
        
        try:
            # 检查元数据兼容性
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True).item()
                if not self._is_index_compatible(metadata):
                    self._logger.warning("索引配置不兼容，需要重新构建")
                    return False
            
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 移动到 GPU（如果配置要求且可用）
            if (self.config.device == "cuda" and 
                hasattr(faiss, 'get_num_gpus') and 
                faiss.get_num_gpus() > 0):
                try:
                    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                    self.index = gpu_index
                    self._logger.info("索引已移动到 GPU")
                except Exception as e:
                    self._logger.warning(f"GPU 索引加载失败，使用 CPU: {e}")
            
            # 加载嵌入
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
            
            self._logger.info(f"FAISS 索引加载成功:")
            self._logger.info(f"  - 向量数量: {self.index.ntotal:,}")
            self._logger.info(f"  - 索引类型: {type(self.index).__name__}")
            self._logger.info(f"  - 内存使用: {self._get_index_memory_usage():.1f} MB")
            
            return True
            
        except Exception as e:
            self._logger.error(f"加载 FAISS 索引失败: {e}")
            return False
    
    def _is_index_compatible(self, metadata: dict) -> bool:
        """检查索引是否与当前配置兼容。
        
        Args:
            metadata: 索引元数据
            
        Returns:
            是否兼容
        """
        # 检查维度
        if metadata.get('dimension') != self.dimension:
            return False
        
        # 检查配置
        config = metadata.get('config', {})
        if config.get('embedding_dim') != self.config.embedding_dim:
            return False
        
        if config.get('normalize_vectors') != self.config.normalize_vectors:
            return False
        
        return True
    
    def get_statistics(self) -> dict:
        """获取索引统计信息。
        
        Returns:
            统计信息字典
        """
        if self.index is None:
            return {}
        
        stats = {
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__,
            'memory_usage_mb': self._get_index_memory_usage()
        }
        
        # 添加特定索引类型的统计
        if hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
        
        if hasattr(self.index, 'hnsw'):
            stats['hnsw_m'] = self.index.hnsw.max_level
        
        return stats
    
    def clear_cache(self) -> None:
        """清理缓存和内存。"""
        self.index = None
        self.embeddings = None
        
        # 清理 GPU 内存（如果使用）
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self._logger.info("向量检索器缓存已清理")


# 导出主要类
__all__ = ['VectorRetriever']