#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Processor Module for RAG System
=======================================

数据集处理器模块，负责加载、预处理和管理文档数据。

Author: Claude AI
Date: 2025-10-29
"""

import os
import pickle
import time
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from loguru import logger

from config import RAGConfig


class DatasetProcessor:
    """数据集处理器，负责加载和预处理文档数据。
    
    主要功能：
    1. 从 HuggingFace 加载数据集
    2. 文档分块处理
    3. 数据清洗和过滤
    4. 缓存管理
    
    Attributes:
        config: RAG系统配置对象
        dataset: 加载的原始数据集
        processed_chunks: 处理后的文档块列表
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化数据集处理器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.processed_chunks: List[Dict[str, Any]] = []
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 确保缓存目录存在
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self._logger.info("数据集处理器初始化完成")
    
    def load_dataset(self) -> None:
        """加载数据集。
        
        从 HuggingFace 加载指定的数据集并进行基本验证。
        
        Raises:
            RuntimeError: 当数据集加载失败时抛出
        """
        try:
            self._logger.info(f"正在加载数据集: {self.config.dataset_name}")
            start_time = time.time()
            
            # 加载数据集
            self.dataset = load_dataset(
                self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            
            # 验证数据集结构
            self._validate_dataset()
            
            # 记录数据集信息
            train_size = len(self.dataset['train']) if 'train' in self.dataset else 0
            self._logger.info(f"数据集加载完成，包含 {train_size:,} 条记录，耗时 {load_time:.2f} 秒")
            
            # 显示数据集示例
            if train_size > 0:
                sample = self.dataset['train'][0]
                self._logger.debug(f"数据集示例: {sample}")
            
        except Exception as e:
            self._logger.error(f"数据集加载失败: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def _validate_dataset(self) -> None:
        """验证数据集结构。
        
        检查数据集是否包含必要的字段和格式。
        
        Raises:
            ValueError: 当数据集格式无效时抛出
        """
        if self.dataset is None:
            raise ValueError("数据集为空")
        
        if 'train' not in self.dataset:
            raise ValueError("数据集缺少 'train' 分割")
        
        train_data = self.dataset['train']
        if len(train_data) == 0:
            raise ValueError("训练数据为空")
        
        # 检查必要字段
        required_fields = ['text']
        sample = train_data[0]
        
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"数据集缺少必要字段: {field}")
        
        self._logger.info("数据集验证通过")
    
    def _chunk_text(self, text: str, title: str = "") -> List[str]:
        """将文本分块处理。
        
        使用智能分块策略，基于句子边界进行分块，保持语义完整性。
        
        Args:
            text: 输入文本
            title: 文档标题（可选）
            
        Returns:
            分块后的文本列表
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # 预处理文本
        text = text.strip()
        
        # 如果文本很短，直接返回
        if len(text) <= self.config.chunk_size:
            return [text]
        
        # 尝试按句子分割
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        # 添加标题到第一个块（如果有）
        if title and title.strip():
            current_chunk = f"{title.strip()}\n\n"
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查添加当前句子是否会超过块大小
            potential_chunk = current_chunk + sentence + " "
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # 当前块已满，保存并开始新块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 处理重叠
                if self.config.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, self.config.chunk_overlap)
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 过滤过短的块
        filtered_chunks = [
            chunk for chunk in chunks 
            if len(chunk.strip()) >= self.config.min_chunk_length
        ]
        
        return filtered_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子。
        
        使用多种句子分割符进行智能分割。
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 句子分割符
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        sentences = [text]
        
        # 逐个使用分割符进行分割
        for ending in sentence_endings:
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split(ending)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:  # 不是最后一部分
                        new_sentences.append(part + ending.strip())
                    else:  # 最后一部分
                        if part.strip():
                            new_sentences.append(part)
            sentences = new_sentences
        
        return [s for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """获取文本的重叠部分。
        
        从文本末尾提取指定长度的重叠文本。
        
        Args:
            text: 源文本
            overlap_size: 重叠大小
            
        Returns:
            重叠文本
        """
        if len(text) <= overlap_size:
            return text
        
        # 尝试在词边界处切割
        overlap_text = text[-overlap_size:]
        space_index = overlap_text.find(' ')
        
        if space_index > 0:
            return overlap_text[space_index + 1:]
        
        return overlap_text
    
    def process_dataset(self) -> None:
        """处理数据集，进行分块和清洗。
        
        将原始数据集转换为适合检索的文档块格式。
        
        Raises:
            ValueError: 当数据集未加载时抛出
        """
        if self.dataset is None:
            raise ValueError("数据集未加载，请先调用 load_dataset()")
        
        self._logger.info("开始处理数据集...")
        start_time = time.time()
        
        # 处理训练集
        train_data = self.dataset['train']
        total_chunks = 0
        processed_docs = 0
        skipped_docs = 0
        
        for idx, item in enumerate(train_data):
            if idx % 1000 == 0:
                self._logger.info(f"已处理 {idx:,}/{len(train_data):,} 篇文档")
            
            try:
                # 获取文本内容和标题
                text_content = item.get('text', '')
                title = item.get('title', f'Document_{idx}')
                
                # 数据清洗
                if not self._is_valid_document(text_content, title):
                    skipped_docs += 1
                    continue
                
                # 分块处理
                chunks = self._chunk_text(text_content, title)
                
                # 保存有效的块
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_data = {
                        'id': f"{idx}_{chunk_idx}",
                        'title': title,
                        'text': chunk,
                        'source_idx': idx,
                        'chunk_idx': chunk_idx,
                        'chunk_length': len(chunk),
                        'original_length': len(text_content)
                    }
                    
                    self.processed_chunks.append(chunk_data)
                    total_chunks += 1
                
                processed_docs += 1
                
            except Exception as e:
                self._logger.warning(f"处理文档 {idx} 时发生错误: {e}")
                skipped_docs += 1
                continue
        
        processing_time = time.time() - start_time
        
        # 记录处理统计
        self._logger.info(f"数据集处理完成:")
        self._logger.info(f"  - 处理文档数: {processed_docs:,}")
        self._logger.info(f"  - 跳过文档数: {skipped_docs:,}")
        self._logger.info(f"  - 生成文档块: {total_chunks:,}")
        self._logger.info(f"  - 平均块长度: {sum(chunk['chunk_length'] for chunk in self.processed_chunks) / len(self.processed_chunks):.1f}")
        self._logger.info(f"  - 处理时间: {processing_time:.2f} 秒")
        
        # 保存处理后的数据
        if self.config.save_intermediate:
            self._save_processed_data()
    
    def _is_valid_document(self, text: str, title: str) -> bool:
        """验证文档是否有效。
        
        Args:
            text: 文档文本
            title: 文档标题
            
        Returns:
            是否为有效文档
        """
        # 检查文本长度
        if not text or len(text.strip()) < self.config.min_chunk_length:
            return False
        
        # 检查文本质量（简单启发式）
        text_clean = text.strip()
        
        # 过滤过短的文档
        if len(text_clean) < 20:
            return False
        
        # 过滤主要是数字或特殊字符的文档
        alpha_ratio = sum(c.isalpha() for c in text_clean) / len(text_clean)
        if alpha_ratio < 0.5:
            return False
        
        # 过滤重复内容过多的文档
        words = text_clean.lower().split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                return False
        
        return True
    
    def _save_processed_data(self) -> None:
        """保存处理后的数据到缓存。"""
        cache_path = os.path.join(self.config.cache_dir, "processed_chunks.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.processed_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._logger.info(f"处理后的数据已保存到: {cache_path}")
            
            # 保存元数据
            metadata = {
                'num_chunks': len(self.processed_chunks),
                'config': self.config.to_dict(),
                'timestamp': time.time()
            }
            
            metadata_path = os.path.join(self.config.cache_dir, "chunks_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            self._logger.error(f"保存处理后的数据失败: {e}")
    
    def load_processed_data(self) -> bool:
        """加载已处理的数据。
        
        Returns:
            是否成功加载缓存数据
        """
        cache_path = os.path.join(self.config.cache_dir, "processed_chunks.pkl")
        metadata_path = os.path.join(self.config.cache_dir, "chunks_metadata.pkl")
        
        if not os.path.exists(cache_path):
            self._logger.info("缓存文件不存在")
            return False
        
        try:
            # 检查元数据兼容性
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # 检查配置兼容性
                if not self._is_cache_compatible(metadata.get('config', {})):
                    self._logger.warning("缓存配置不兼容，需要重新处理")
                    return False
            
            # 加载处理后的数据
            with open(cache_path, 'rb') as f:
                self.processed_chunks = pickle.load(f)
            
            self._logger.info(f"从缓存加载了 {len(self.processed_chunks):,} 个文档块")
            
            # 显示统计信息
            if self.processed_chunks:
                avg_length = sum(chunk['chunk_length'] for chunk in self.processed_chunks) / len(self.processed_chunks)
                self._logger.info(f"平均块长度: {avg_length:.1f}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"加载缓存数据失败: {e}")
            return False
    
    def _is_cache_compatible(self, cached_config: Dict[str, Any]) -> bool:
        """检查缓存配置是否与当前配置兼容。
        
        Args:
            cached_config: 缓存的配置字典
            
        Returns:
            是否兼容
        """
        # 检查关键配置参数
        key_params = [
            'dataset_name', 'chunk_size', 'chunk_overlap', 
            'min_chunk_length'
        ]
        
        current_config = self.config.to_dict()
        
        for param in key_params:
            if cached_config.get(param) != current_config.get(param):
                self._logger.debug(f"配置参数 {param} 不匹配: {cached_config.get(param)} vs {current_config.get(param)}")
                return False
        
        return True
    
    def get_processed_chunks(self) -> List[Dict[str, Any]]:
        """获取处理后的文档块。
        
        Returns:
            处理后的文档块列表
        """
        return self.processed_chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集处理统计信息。
        
        Returns:
            统计信息字典
        """
        if not self.processed_chunks:
            return {}
        
        chunk_lengths = [chunk['chunk_length'] for chunk in self.processed_chunks]
        
        return {
            'total_chunks': len(self.processed_chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_text_length': sum(chunk_lengths),
            'unique_documents': len(set(chunk['source_idx'] for chunk in self.processed_chunks))
        }
    
    def clear_cache(self) -> None:
        """清空缓存数据。"""
        cache_files = [
            "processed_chunks.pkl",
            "chunks_metadata.pkl"
        ]
        
        cleared_count = 0
        for filename in cache_files:
            filepath = os.path.join(self.config.cache_dir, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    cleared_count += 1
                except Exception as e:
                    self._logger.error(f"清除缓存文件 {filepath} 失败: {e}")
        
        self.processed_chunks = []
        self._logger.info(f"已清除 {cleared_count} 个缓存文件")


# 导出主要类
__all__ = ['DatasetProcessor']