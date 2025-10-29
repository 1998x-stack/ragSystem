#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Retriever Module for RAG System
======================================

混合检索器模块，结合向量检索和BM25检索策略。

Author: Claude AI
Date: 2025-10-29
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from bm25_retriever import BM25Retriever
from config import RAGConfig
from embedding_processor import EmbeddingProcessor
from vector_retriever import VectorRetriever


class HybridRetriever:
    """混合检索器，结合向量检索和BM25检索。
    
    主要功能：
    1. 协调向量检索和BM25检索
    2. 结果融合和重排序
    3. 分数标准化和加权
    4. 结果去重和过滤
    
    Attributes:
        config: RAG系统配置对象
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        embedding_processor: 嵌入处理器
        documents: 文档列表
        _logger: 日志记录器
    """
    
    def __init__(
        self,
        config: RAGConfig,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        embedding_processor: EmbeddingProcessor
    ) -> None:
        """初始化混合检索器。
        
        Args:
            config: RAG系统配置对象
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            embedding_processor: 嵌入处理器
        """
        self.config = config
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.embedding_processor = embedding_processor
        self.documents: List[Dict[str, Any]] = []
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 验证组件
        self._validate_components()
        
        self._logger.info("混合检索器初始化完成")
        self._logger.info(f"检索权重配置: 向量={config.embedding_weight:.2f}, BM25={config.bm25_weight:.2f}")
    
    def _validate_components(self) -> None:
        """验证检索组件的有效性。
        
        Raises:
            ValueError: 当组件配置无效时抛出
        """
        if self.vector_retriever is None:
            raise ValueError("向量检索器不能为None")
        
        if self.bm25_retriever is None:
            raise ValueError("BM25检索器不能为None")
        
        if self.embedding_processor is None:
            raise ValueError("嵌入处理器不能为None")
        
        # 检查权重配置
        total_weight = self.config.embedding_weight + self.config.bm25_weight
        if abs(total_weight - 1.0) > 1e-6:
            self._logger.warning(f"检索权重总和不为1: {total_weight:.4f}")
    
    def set_documents(self, documents: List[Dict[str, Any]]) -> None:
        """设置文档列表。
        
        Args:
            documents: 文档列表，每个文档应包含id、title、text等字段
            
        Raises:
            ValueError: 当文档列表格式无效时抛出
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        # 验证文档格式
        required_fields = {'id', 'text'}
        for i, doc in enumerate(documents[:10]):  # 检查前10个文档
            if not isinstance(doc, dict):
                raise ValueError(f"文档 {i} 不是字典格式")
            
            missing_fields = required_fields - set(doc.keys())
            if missing_fields:
                raise ValueError(f"文档 {i} 缺少必要字段: {missing_fields}")
        
        self.documents = documents
        self._logger.info(f"设置了 {len(documents):,} 个文档")
        
        # 显示文档统计信息
        self._log_document_statistics()
    
    def _log_document_statistics(self) -> None:
        """记录文档统计信息。"""
        if not self.documents:
            return
        
        # 计算文档长度统计
        text_lengths = [len(doc.get('text', '')) for doc in self.documents]
        
        stats = {
            'total_docs': len(self.documents),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'has_title': sum(1 for doc in self.documents if doc.get('title'))
        }
        
        self._logger.info(f"文档统计:")
        self._logger.info(f"  - 文档数量: {stats['total_docs']:,}")
        self._logger.info(f"  - 平均文本长度: {stats['avg_text_length']:.1f}")
        self._logger.info(f"  - 文本长度范围: {stats['min_text_length']}-{stats['max_text_length']}")
        self._logger.info(f"  - 包含标题的文档: {stats['has_title']:,}")
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        retrieval_strategy: str = "hybrid",
        rerank_top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """执行混合搜索。
        
        Args:
            query: 查询字符串
            k: 返回的文档数量，默认使用配置中的值
            retrieval_strategy: 检索策略 ("hybrid", "vector_only", "bm25_only")
            rerank_top_k: 重排序前保留的候选数量
            
        Returns:
            检索到的文档列表，按相关性排序
            
        Raises:
            ValueError: 当查询为空或文档未设置时抛出
        """
        if not query.strip():
            raise ValueError("查询不能为空")
        
        if not self.documents:
            raise ValueError("文档未设置，请先调用 set_documents()")
        
        k = k or self.config.top_k_retrieval
        rerank_top_k = rerank_top_k or k * 3  # 候选数量是最终结果的3倍
        
        self._logger.info(f"执行{retrieval_strategy}检索: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        start_time = time.time()
        
        try:
            if retrieval_strategy == "vector_only":
                results = self._vector_only_search(query, k)
            elif retrieval_strategy == "bm25_only":
                results = self._bm25_only_search(query, k)
            else:  # hybrid
                results = self._hybrid_search(query, k, rerank_top_k)
            
            search_time = time.time() - start_time
            
            self._logger.info(f"检索完成，返回 {len(results)} 个结果，耗时 {search_time:.3f} 秒")
            
            return results
            
        except Exception as e:
            self._logger.error(f"检索过程中发生错误: {e}")
            raise
    
    def _hybrid_search(self, query: str, k: int, rerank_top_k: int) -> List[Dict[str, Any]]:
        """执行混合检索。
        
        Args:
            query: 查询字符串
            k: 最终返回数量
            rerank_top_k: 候选数量
            
        Returns:
            检索结果列表
        """
        # 1. 向量检索
        vector_scores, vector_indices = self._perform_vector_search(query, rerank_top_k)
        
        # 2. BM25检索
        bm25_scores, bm25_indices = self._perform_bm25_search(query, rerank_top_k)
        
        # 3. 合并和融合结果
        candidates = self._merge_retrieval_results(
            vector_scores, vector_indices,
            bm25_scores, bm25_indices
        )
        
        # 4. 重排序和过滤
        final_results = self._rank_and_filter_candidates(candidates, k)
        
        return final_results
    
    def _vector_only_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """仅使用向量检索。
        
        Args:
            query: 查询字符串
            k: 返回数量
            
        Returns:
            检索结果列表
        """
        vector_scores, vector_indices = self._perform_vector_search(query, k)
        
        results = []
        for score, idx in zip(vector_scores, vector_indices):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['retrieval_score'] = float(score)
                doc['vector_score'] = float(score)
                doc['bm25_score'] = 0.0
                doc['retrieval_method'] = 'vector_only'
                results.append(doc)
        
        return results
    
    def _bm25_only_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """仅使用BM25检索。
        
        Args:
            query: 查询字符串
            k: 返回数量
            
        Returns:
            检索结果列表
        """
        bm25_scores, bm25_indices = self._perform_bm25_search(query, k)
        
        results = []
        for score, idx in zip(bm25_scores, bm25_indices):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['retrieval_score'] = float(score)
                doc['vector_score'] = 0.0
                doc['bm25_score'] = float(score)
                doc['retrieval_method'] = 'bm25_only'
                results.append(doc)
        
        return results
    
    def _perform_vector_search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """执行向量检索。
        
        Args:
            query: 查询字符串
            k: 检索数量
            
        Returns:
            (向量分数, 向量索引)
        """
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_processor.encode_texts([query])
            
            # 向量检索
            scores, indices = self.vector_retriever.search(query_embedding, k)
            
            self._logger.debug(f"向量检索完成，返回 {len(scores)} 个结果")
            
            return scores, indices
            
        except Exception as e:
            self._logger.error(f"向量检索失败: {e}")
            return np.array([]), np.array([])
    
    def _perform_bm25_search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """执行BM25检索。
        
        Args:
            query: 查询字符串
            k: 检索数量
            
        Returns:
            (BM25分数, BM25索引)
        """
        try:
            scores, indices = self.bm25_retriever.search(query, k)
            
            self._logger.debug(f"BM25检索完成，返回 {len(scores)} 个结果")
            
            return scores, indices
            
        except Exception as e:
            self._logger.error(f"BM25检索失败: {e}")
            return np.array([]), np.array([])
    
    def _merge_retrieval_results(
        self,
        vector_scores: np.ndarray,
        vector_indices: np.ndarray,
        bm25_scores: np.ndarray,
        bm25_indices: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """合并检索结果。
        
        Args:
            vector_scores: 向量检索分数
            vector_indices: 向量检索索引
            bm25_scores: BM25检索分数
            bm25_indices: BM25检索索引
            
        Returns:
            候选文档字典
        """
        candidates = {}
        
        # 标准化分数
        norm_vector_scores = self._normalize_scores(vector_scores)
        norm_bm25_scores = self._normalize_scores(bm25_scores)
        
        # 添加向量检索结果
        for i, (norm_score, raw_score, idx) in enumerate(zip(norm_vector_scores, vector_scores, vector_indices)):
            if idx < len(self.documents):
                candidates[idx] = {
                    'doc_index': idx,
                    'vector_score': float(raw_score),
                    'bm25_score': 0.0,
                    'vector_norm_score': float(norm_score),
                    'bm25_norm_score': 0.0,
                    'vector_rank': i,
                    'bm25_rank': float('inf')
                }
        
        # 添加BM25检索结果
        for i, (norm_score, raw_score, idx) in enumerate(zip(norm_bm25_scores, bm25_scores, bm25_indices)):
            if idx < len(self.documents):
                if idx in candidates:
                    candidates[idx]['bm25_score'] = float(raw_score)
                    candidates[idx]['bm25_norm_score'] = float(norm_score)
                    candidates[idx]['bm25_rank'] = i
                else:
                    candidates[idx] = {
                        'doc_index': idx,
                        'vector_score': 0.0,
                        'bm25_score': float(raw_score),
                        'vector_norm_score': 0.0,
                        'bm25_norm_score': float(norm_score),
                        'vector_rank': float('inf'),
                        'bm25_rank': i
                    }
        
        # 计算组合分数
        for candidate in candidates.values():
            candidate['combined_score'] = self._calculate_combined_score(candidate)
        
        self._logger.debug(f"合并结果，共 {len(candidates)} 个候选文档")
        
        return candidates
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """标准化分数到[0,1]范围。
        
        Args:
            scores: 原始分数数组
            
        Returns:
            标准化后的分数数组
        """
        if len(scores) == 0:
            return scores
        
        if len(scores) == 1:
            return np.array([1.0])
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score == 0:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _calculate_combined_score(self, candidate: Dict[str, Any]) -> float:
        """计算组合分数。
        
        Args:
            candidate: 候选文档信息
            
        Returns:
            组合分数
        """
        # 基于标准化分数的加权组合
        vector_contrib = self.config.embedding_weight * candidate['vector_norm_score']
        bm25_contrib = self.config.bm25_weight * candidate['bm25_norm_score']
        
        # 添加排名奖励（排名越靠前，奖励越高）
        vector_rank_bonus = 0.0
        if candidate['vector_rank'] != float('inf'):
            vector_rank_bonus = self.config.embedding_weight * 0.1 / (candidate['vector_rank'] + 1)
        
        bm25_rank_bonus = 0.0
        if candidate['bm25_rank'] != float('inf'):
            bm25_rank_bonus = self.config.bm25_weight * 0.1 / (candidate['bm25_rank'] + 1)
        
        combined_score = vector_contrib + bm25_contrib + vector_rank_bonus + bm25_rank_bonus
        
        return combined_score
    
    def _rank_and_filter_candidates(
        self,
        candidates: Dict[int, Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """对候选文档进行排序和过滤。
        
        Args:
            candidates: 候选文档字典
            k: 最终返回数量
            
        Returns:
            排序后的文档列表
        """
        # 按组合分数排序
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:k]
        
        # 构建最终结果
        results = []
        for candidate in sorted_candidates:
            doc_idx = candidate['doc_index']
            doc = self.documents[doc_idx].copy()
            
            # 添加检索元信息
            doc.update({
                'retrieval_score': candidate['combined_score'],
                'vector_score': candidate['vector_score'],
                'bm25_score': candidate['bm25_score'],
                'vector_rank': candidate['vector_rank'] if candidate['vector_rank'] != float('inf') else None,
                'bm25_rank': candidate['bm25_rank'] if candidate['bm25_rank'] != float('inf') else None,
                'retrieval_method': 'hybrid'
            })
            
            results.append(doc)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        k: Optional[int] = None,
        retrieval_strategy: str = "hybrid"
    ) -> List[List[Dict[str, Any]]]:
        """批量检索。
        
        Args:
            queries: 查询列表
            k: 每个查询返回的文档数量
            retrieval_strategy: 检索策略
            
        Returns:
            每个查询的检索结果列表
        """
        results = []
        
        self._logger.info(f"批量检索 {len(queries)} 个查询")
        start_time = time.time()
        
        for i, query in enumerate(queries):
            if i % 10 == 0 and i > 0:
                self._logger.debug(f"批量检索进度: {i}/{len(queries)}")
            
            try:
                query_results = self.search(query, k, retrieval_strategy)
                results.append(query_results)
            except Exception as e:
                self._logger.error(f"查询 {i} 检索失败: {e}")
                results.append([])
        
        total_time = time.time() - start_time
        self._logger.info(f"批量检索完成，耗时 {total_time:.2f} 秒")
        
        return results
    
    def explain_retrieval(self, query: str, doc_index: int) -> Dict[str, Any]:
        """解释检索评分的计算过程。
        
        Args:
            query: 查询字符串
            doc_index: 文档索引
            
        Returns:
            评分解释字典
        """
        if doc_index >= len(self.documents):
            return {}
        
        explanation = {
            'document': self.documents[doc_index],
            'query': query,
            'vector_explanation': {},
            'bm25_explanation': {},
            'hybrid_scores': {}
        }
        
        try:
            # 向量检索解释
            query_embedding = self.embedding_processor.encode_texts([query])
            vector_scores, vector_indices = self.vector_retriever.search(query_embedding, len(self.documents))
            
            if doc_index in vector_indices:
                pos = np.where(vector_indices == doc_index)[0][0]
                explanation['vector_explanation'] = {
                    'score': float(vector_scores[pos]),
                    'rank': int(pos),
                    'method': 'cosine_similarity'
                }
            
            # BM25检索解释
            explanation['bm25_explanation'] = self.bm25_retriever.explain_score(query, doc_index)
            
            # 混合分数计算
            if explanation['vector_explanation'] and explanation['bm25_explanation']:
                vector_score = explanation['vector_explanation']['score']
                bm25_score = explanation['bm25_explanation']['total_score']
                
                combined_score = (
                    self.config.embedding_weight * vector_score +
                    self.config.bm25_weight * bm25_score
                )
                
                explanation['hybrid_scores'] = {
                    'vector_weighted': self.config.embedding_weight * vector_score,
                    'bm25_weighted': self.config.bm25_weight * bm25_score,
                    'combined': combined_score,
                    'weights': {
                        'vector': self.config.embedding_weight,
                        'bm25': self.config.bm25_weight
                    }
                }
            
        except Exception as e:
            self._logger.error(f"检索解释失败: {e}")
            explanation['error'] = str(e)
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取混合检索器统计信息。
        
        Returns:
            统计信息字典
        """
        stats = {
            'num_documents': len(self.documents),
            'retrieval_weights': {
                'vector': self.config.embedding_weight,
                'bm25': self.config.bm25_weight
            },
            'vector_retriever_stats': self.vector_retriever.get_statistics(),
            'bm25_retriever_stats': self.bm25_retriever.get_statistics()
        }
        
        return stats
    
    def clear_cache(self) -> None:
        """清理缓存。"""
        self.documents = []
        self._logger.info("混合检索器缓存已清理")


# 导出主要类
__all__ = ['HybridRetriever']