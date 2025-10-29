#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25 Retriever Module for RAG System
====================================

BM25检索器模块，实现基于关键词的文档检索功能。

Author: Claude AI
Date: 2025-10-29
"""

import os
import pickle
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from config import RAGConfig


class BM25Retriever:
    """BM25检索器，进行关键词匹配和相关性计算。
    
    主要功能：
    1. 构建和管理BM25索引
    2. 文本预处理和分词
    3. 关键词检索和相关性评分
    4. 索引持久化和缓存管理
    
    Attributes:
        config: RAG系统配置对象
        bm25: BM25模型实例
        corpus: 文档语料库
        tokenized_corpus: 分词后的语料库
        stopwords: 停用词集合
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化BM25检索器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.stopwords: Set[str] = set()
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 加载停用词
        self._load_stopwords()
        
        self._logger.info("BM25检索器初始化完成")
    
    def _load_stopwords(self) -> None:
        """加载停用词列表。"""
        try:
            # 尝试加载NLTK停用词
            import nltk
            try:
                from nltk.corpus import stopwords
                # 英文停用词
                english_stopwords = set(stopwords.words('english'))
                self.stopwords.update(english_stopwords)
                self._logger.debug(f"加载了 {len(english_stopwords)} 个英文停用词")
            except LookupError:
                self._logger.warning("NLTK stopwords 数据未找到，使用内置停用词")
                
        except ImportError:
            self._logger.info("NLTK 未安装，使用内置停用词")
        
        # 内置英文停用词
        builtin_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'i', 'you', 'we', 'they',
            'this', 'these', 'those', 'but', 'or', 'not', 'can', 'could',
            'should', 'have', 'had', 'do', 'does', 'did', 'been', 'being'
        }
        self.stopwords.update(builtin_stopwords)
        
        # 常见中文停用词
        chinese_stopwords = {
            '的', '了', '是', '在', '有', '和', '就', '不', '与', '也', '都',
            '要', '可以', '这', '那', '没有', '很', '还', '会', '上', '下',
            '来', '去', '一个', '我们', '他们', '她们', '什么', '怎么', '为什么'
        }
        self.stopwords.update(chinese_stopwords)
        
        self._logger.info(f"停用词加载完成，共 {len(self.stopwords)} 个")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """文本分词处理。
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留字母、数字和中文字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 基本分词（按空格分割）
        tokens = text.split()
        
        # 处理中英文混合文本
        processed_tokens = []
        for token in tokens:
            # 进一步处理包含中文的token
            if re.search(r'[\u4e00-\u9fff]', token):
                # 中文字符级分词
                processed_tokens.extend(list(token))
            else:
                # 英文单词
                processed_tokens.append(token)
        
        # 过滤停用词和短token
        filtered_tokens = [
            token for token in processed_tokens
            if len(token) > 1 and token not in self.stopwords
        ]
        
        return filtered_tokens
    
    def _preprocess_corpus(self, documents: List[str]) -> List[List[str]]:
        """预处理文档语料库。
        
        Args:
            documents: 文档列表
            
        Returns:
            分词后的文档列表
        """
        self._logger.info(f"预处理 {len(documents):,} 个文档")
        start_time = time.time()
        
        tokenized_docs = []
        
        for i, doc in enumerate(documents):
            if i % 10000 == 0 and i > 0:
                self._logger.debug(f"预处理进度: {i:,}/{len(documents):,}")
            
            tokens = self._tokenize_text(doc)
            tokenized_docs.append(tokens)
        
        # 统计信息
        total_tokens = sum(len(tokens) for tokens in tokenized_docs)
        avg_tokens = total_tokens / len(tokenized_docs) if tokenized_docs else 0
        
        processing_time = time.time() - start_time
        
        self._logger.info(f"文档预处理完成:")
        self._logger.info(f"  - 总token数: {total_tokens:,}")
        self._logger.info(f"  - 平均token数: {avg_tokens:.1f}")
        self._logger.info(f"  - 处理时间: {processing_time:.2f} 秒")
        
        return tokenized_docs
    
    def build_index(self, documents: List[str]) -> None:
        """构建BM25索引。
        
        Args:
            documents: 文档列表
            
        Raises:
            ValueError: 当文档列表为空时抛出
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        self._logger.info("开始构建BM25索引...")
        start_time = time.time()
        
        # 保存原始语料库
        self.corpus = documents.copy()
        
        # 预处理和分词
        self.tokenized_corpus = self._preprocess_corpus(documents)
        
        # 过滤空文档
        non_empty_docs = [doc for doc in self.tokenized_corpus if doc]
        if len(non_empty_docs) < len(self.tokenized_corpus):
            empty_count = len(self.tokenized_corpus) - len(non_empty_docs)
            self._logger.warning(f"过滤了 {empty_count} 个空文档")
            self.tokenized_corpus = non_empty_docs
        
        # 构建BM25模型
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        
        build_time = time.time() - start_time
        
        # 计算索引统计信息
        vocab_size = len(set(token for doc in self.tokenized_corpus for token in doc))
        
        self._logger.info(f"BM25索引构建完成:")
        self._logger.info(f"  - 文档数量: {len(self.tokenized_corpus):,}")
        self._logger.info(f"  - 词汇量: {vocab_size:,}")
        self._logger.info(f"  - 构建时间: {build_time:.2f} 秒")
        self._logger.info(f"  - BM25参数: k1={self.config.bm25_k1}, b={self.config.bm25_b}")
    
    def search(
        self, 
        query: str, 
        k: int,
        min_score: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相关的文档。
        
        Args:
            query: 查询字符串
            k: 返回的文档数量
            min_score: 最小分数阈值
            
        Returns:
            (BM25分数数组, 文档索引数组)
            
        Raises:
            ValueError: 当索引未构建时抛出
        """
        if self.bm25 is None:
            raise ValueError("BM25索引未构建，请先调用 build_index()")
        
        if not query.strip():
            return np.array([]), np.array([])
        
        # 预处理查询
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            self._logger.warning(f"查询预处理后为空: '{query}'")
            return np.array([]), np.array([])
        
        self._logger.debug(f"查询tokens: {query_tokens}")
        
        # 获取所有文档的BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 应用最小分数阈值
        valid_indices = np.where(scores >= min_score)[0]
        valid_scores = scores[valid_indices]
        
        # 获取top-k结果
        if len(valid_scores) > k:
            top_k_positions = np.argpartition(valid_scores, -k)[-k:]
            top_k_positions = top_k_positions[np.argsort(valid_scores[top_k_positions])[::-1]]
            
            top_indices = valid_indices[top_k_positions]
            top_scores = valid_scores[top_k_positions]
        else:
            # 按分数排序
            sorted_positions = np.argsort(valid_scores)[::-1]
            top_indices = valid_indices[sorted_positions]
            top_scores = valid_scores[sorted_positions]
        
        self._logger.debug(f"BM25搜索完成，返回 {len(top_scores)} 个结果")
        
        return top_scores, top_indices
    
    def batch_search(
        self, 
        queries: List[str], 
        k: int,
        min_score: float = 0.0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """批量搜索。
        
        Args:
            queries: 查询列表
            k: 每个查询返回的文档数量
            min_score: 最小分数阈值
            
        Returns:
            每个查询的(分数, 索引)结果列表
        """
        results = []
        
        self._logger.info(f"批量搜索 {len(queries)} 个查询")
        
        for i, query in enumerate(queries):
            if i % 100 == 0 and i > 0:
                self._logger.debug(f"批量搜索进度: {i}/{len(queries)}")
            
            scores, indices = self.search(query, k, min_score)
            results.append((scores, indices))
        
        return results
    
    def get_document_by_index(self, index: int) -> Optional[str]:
        """根据索引获取原始文档。
        
        Args:
            index: 文档索引
            
        Returns:
            原始文档文本，如果索引无效则返回None
        """
        if 0 <= index < len(self.corpus):
            return self.corpus[index]
        return None
    
    def get_tokenized_document(self, index: int) -> Optional[List[str]]:
        """根据索引获取分词后的文档。
        
        Args:
            index: 文档索引
            
        Returns:
            分词后的文档，如果索引无效则返回None
        """
        if 0 <= index < len(self.tokenized_corpus):
            return self.tokenized_corpus[index]
        return None
    
    def explain_score(self, query: str, doc_index: int) -> Dict[str, any]:
        """解释BM25评分的计算过程。
        
        Args:
            query: 查询字符串
            doc_index: 文档索引
            
        Returns:
            评分解释字典
        """
        if self.bm25 is None or doc_index >= len(self.tokenized_corpus):
            return {}
        
        query_tokens = self._tokenize_text(query)
        doc_tokens = self.tokenized_corpus[doc_index]
        
        explanation = {
            'query_tokens': query_tokens,
            'doc_length': len(doc_tokens),
            'avg_doc_length': self.bm25.avgdl,
            'doc_tokens': doc_tokens,
            'term_scores': {},
            'total_score': 0.0
        }
        
        # 计算每个查询词的贡献
        for token in query_tokens:
            if token in doc_tokens:
                # 计算词频
                tf = doc_tokens.count(token)
                
                # 计算IDF
                doc_freq = sum(1 for doc in self.tokenized_corpus if token in doc)
                idf = np.log((len(self.tokenized_corpus) - doc_freq + 0.5) / (doc_freq + 0.5))
                
                # 计算BM25分数
                score = idf * (tf * (self.config.bm25_k1 + 1)) / (
                    tf + self.config.bm25_k1 * (
                        1 - self.config.bm25_b + 
                        self.config.bm25_b * len(doc_tokens) / self.bm25.avgdl
                    )
                )
                
                explanation['term_scores'][token] = {
                    'tf': tf,
                    'idf': idf,
                    'score': score
                }
                explanation['total_score'] += score
        
        return explanation
    
    def get_vocabulary_stats(self) -> Dict[str, any]:
        """获取词汇统计信息。
        
        Returns:
            词汇统计字典
        """
        if not self.tokenized_corpus:
            return {}
        
        # 统计词汇
        word_freq = {}
        total_tokens = 0
        
        for doc in self.tokenized_corpus:
            total_tokens += len(doc)
            for token in doc:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # 计算统计信息
        vocab_size = len(word_freq)
        most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'vocab_size': vocab_size,
            'total_tokens': total_tokens,
            'avg_tokens_per_doc': total_tokens / len(self.tokenized_corpus),
            'most_common_words': most_common,
            'singleton_words': sum(1 for freq in word_freq.values() if freq == 1)
        }
    
    def save_index(self, filename: str = "bm25_index.pkl") -> None:
        """保存BM25索引。
        
        Args:
            filename: 文件名
        """
        if self.bm25 is None:
            self._logger.warning("BM25索引未构建，无法保存")
            return
        
        filepath = os.path.join(self.config.cache_dir, filename)
        
        try:
            # 确保目录存在
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
            # 准备保存数据
            save_data = {
                'bm25': self.bm25,
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                'stopwords': self.stopwords,
                'config': {
                    'bm25_k1': self.config.bm25_k1,
                    'bm25_b': self.config.bm25_b
                },
                'timestamp': time.time(),
                'vocabulary_stats': self.get_vocabulary_stats()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 计算文件大小
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            
            self._logger.info(f"BM25索引已保存:")
            self._logger.info(f"  - 文件路径: {filepath}")
            self._logger.info(f"  - 文件大小: {file_size:.1f} MB")
            self._logger.info(f"  - 文档数量: {len(self.corpus):,}")
            
        except Exception as e:
            self._logger.error(f"保存BM25索引失败: {e}")
    
    def load_index(self, filename: str = "bm25_index.pkl") -> bool:
        """加载BM25索引。
        
        Args:
            filename: 文件名
            
        Returns:
            是否成功加载
        """
        filepath = os.path.join(self.config.cache_dir, filename)
        
        if not os.path.exists(filepath):
            self._logger.info(f"BM25索引文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # 检查兼容性
            if not self._is_index_compatible(save_data.get('config', {})):
                self._logger.warning("BM25索引配置不兼容，需要重新构建")
                return False
            
            # 加载数据
            self.bm25 = save_data['bm25']
            self.corpus = save_data['corpus']
            self.tokenized_corpus = save_data['tokenized_corpus']
            self.stopwords = save_data.get('stopwords', self.stopwords)
            
            # 显示加载信息
            vocab_stats = save_data.get('vocabulary_stats', {})
            
            self._logger.info(f"BM25索引加载成功:")
            self._logger.info(f"  - 文档数量: {len(self.corpus):,}")
            self._logger.info(f"  - 词汇量: {vocab_stats.get('vocab_size', 'N/A'):,}")
            self._logger.info(f"  - 平均文档长度: {vocab_stats.get('avg_tokens_per_doc', 'N/A'):.1f}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"加载BM25索引失败: {e}")
            return False
    
    def _is_index_compatible(self, saved_config: Dict[str, any]) -> bool:
        """检查索引是否与当前配置兼容。
        
        Args:
            saved_config: 保存的配置
            
        Returns:
            是否兼容
        """
        # 检查关键参数
        key_params = ['bm25_k1', 'bm25_b']
        
        for param in key_params:
            if saved_config.get(param) != getattr(self.config, param):
                self._logger.debug(f"BM25参数不匹配: {param}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, any]:
        """获取BM25检索器统计信息。
        
        Returns:
            统计信息字典
        """
        if not self.corpus:
            return {}
        
        stats = {
            'num_documents': len(self.corpus),
            'num_tokenized_documents': len(self.tokenized_corpus),
            'total_stopwords': len(self.stopwords),
            'bm25_k1': self.config.bm25_k1,
            'bm25_b': self.config.bm25_b
        }
        
        # 添加词汇统计
        vocab_stats = self.get_vocabulary_stats()
        stats.update(vocab_stats)
        
        # 添加平均文档长度
        if self.bm25:
            stats['avg_doc_length'] = self.bm25.avgdl
        
        return stats
    
    def clear_cache(self) -> None:
        """清理缓存和内存。"""
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        
        self._logger.info("BM25检索器缓存已清理")


# 导出主要类
__all__ = ['BM25Retriever']