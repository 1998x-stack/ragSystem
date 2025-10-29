#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline Module for RAG System
==================================

RAG主流水线模块，协调整个检索增强生成过程。

Author: Claude AI
Date: 2025-10-29
"""

import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from bm25_retriever import BM25Retriever
from config import RAGConfig
from context_organizer import ContextOrganizer
from dataset_processor import DatasetProcessor
from embedding_processor import EmbeddingProcessor
from hybrid_retriever import HybridRetriever
from query_processor import QueryProcessor
from vector_retriever import VectorRetriever


class RAGPipeline:
    """RAG主流水线，协调整个检索增强生成过程。
    
    主要功能：
    1. 系统初始化和组件协调
    2. 完整的RAG查询处理流程
    3. 性能监控和错误处理
    4. 系统状态管理和优化
    
    Attributes:
        config: RAG系统配置对象
        dataset_processor: 数据集处理器
        embedding_processor: 嵌入处理器
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        hybrid_retriever: 混合检索器
        query_processor: 查询处理器
        context_organizer: 上下文组织器
        is_initialized: 是否已初始化
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化RAG流水线。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 初始化组件
        self.dataset_processor = DatasetProcessor(config)
        self.embedding_processor = EmbeddingProcessor(config)
        self.vector_retriever = VectorRetriever(config)
        self.bm25_retriever = BM25Retriever(config)
        self.hybrid_retriever = HybridRetriever(
            config,
            self.vector_retriever,
            self.bm25_retriever,
            self.embedding_processor
        )
        self.query_processor = QueryProcessor(config)
        self.context_organizer = ContextOrganizer(config)
        
        # 系统状态
        self.is_initialized = False
        self._initialization_time = 0.0
        self._query_count = 0
        self._total_query_time = 0.0
        
        self._logger.info("RAG流水线初始化完成")
        self._logger.info(f"配置摘要: 嵌入维度={config.embedding_dim}, 检索数量={config.top_k_retrieval}")
    
    def initialize(
        self,
        force_rebuild: bool = False,
        enable_gpu: bool = None,
        show_progress: bool = True
    ) -> None:
        """初始化RAG系统。
        
        Args:
            force_rebuild: 是否强制重建索引
            enable_gpu: 是否启用GPU（None表示自动检测）
            show_progress: 是否显示进度信息
            
        Raises:
            RuntimeError: 当初始化失败时抛出
        """
        if self.is_initialized and not force_rebuild:
            self._logger.info("系统已初始化，跳过初始化过程")
            return
        
        self._logger.info("开始初始化RAG系统...")
        start_time = time.time()
        
        try:
            # 步骤1：检查和调整配置
            self._adjust_config_for_environment(enable_gpu)
            
            # 步骤2：加载和处理数据集
            self._initialize_dataset(force_rebuild, show_progress)
            
            # 步骤3：加载模型
            self._initialize_models(show_progress)
            
            # 步骤4：构建索引
            self._initialize_indexes(force_rebuild, show_progress)
            
            # 步骤5：设置检索器
            self._initialize_retrievers(show_progress)
            
            # 步骤6：系统验证
            self._validate_system(show_progress)
            
            self._initialization_time = time.time() - start_time
            self.is_initialized = True
            
            self._logger.info(f"RAG系统初始化完成，总耗时: {self._initialization_time:.2f} 秒")
            self._log_initialization_summary()
            
        except Exception as e:
            self._logger.error(f"RAG系统初始化失败: {e}")
            self.is_initialized = False
            raise RuntimeError(f"RAG system initialization failed: {e}")
    
    def _adjust_config_for_environment(self, enable_gpu: Optional[bool]) -> None:
        """根据环境调整配置。
        
        Args:
            enable_gpu: 是否启用GPU
        """
        self._logger.info("检查系统环境...")
        
        # GPU设置
        if enable_gpu is not None:
            if enable_gpu and not torch.cuda.is_available():
                self._logger.warning("请求使用GPU但CUDA不可用，使用CPU")
                self.config.device = "cpu"
            elif enable_gpu:
                self.config.device = "cuda"
            else:
                self.config.device = "cpu"
        
        # 内存优化
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self._logger.info(f"GPU内存: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                self._logger.warning("GPU内存较少，自动调整配置")
                self.config.embedding_dim = min(self.config.embedding_dim, 512)
                self.config.batch_size = min(self.config.batch_size, 16)
        
        self._logger.info(f"使用设备: {self.config.device}")
        self._logger.info(f"配置调整: 嵌入维度={self.config.embedding_dim}, 批大小={self.config.batch_size}")
    
    def _initialize_dataset(self, force_rebuild: bool, show_progress: bool) -> None:
        """初始化数据集。
        
        Args:
            force_rebuild: 是否强制重建
            show_progress: 是否显示进度
        """
        if show_progress:
            self._logger.info("步骤1/6: 初始化数据集...")
        
        # 尝试加载缓存的数据
        if not force_rebuild and self.dataset_processor.load_processed_data():
            self._logger.info("已加载缓存的数据集")
        else:
            # 加载和处理数据集
            self.dataset_processor.load_dataset()
            self.dataset_processor.process_dataset()
        
        # 验证数据集
        chunks = self.dataset_processor.get_processed_chunks()
        if not chunks:
            raise RuntimeError("没有可用的文档块")
        
        self._logger.info(f"数据集就绪，文档块数量: {len(chunks):,}")
    
    def _initialize_models(self, show_progress: bool) -> None:
        """初始化模型。
        
        Args:
            show_progress: 是否显示进度
        """
        if show_progress:
            self._logger.info("步骤2/6: 加载模型...")
        
        # 加载嵌入模型
        self.embedding_processor.load_model()
        
        # 加载语言模型
        self.query_processor.load_model()
        
        self._logger.info("所有模型加载完成")
    
    def _initialize_indexes(self, force_rebuild: bool, show_progress: bool) -> None:
        """初始化索引。
        
        Args:
            force_rebuild: 是否强制重建
            show_progress: 是否显示进度
        """
        chunks = self.dataset_processor.get_processed_chunks()
        texts = [chunk['text'] for chunk in chunks]
        
        # 初始化向量索引
        if show_progress:
            self._logger.info("步骤3/6: 构建向量索引...")
        
        if not force_rebuild and self.vector_retriever.load_index():
            self._logger.info("已加载缓存的向量索引")
        else:
            # 生成嵌入
            embeddings = self.embedding_processor.encode_texts(texts, show_progress=show_progress)
            self.embedding_processor.save_embeddings(embeddings)
            
            # 构建向量索引
            self.vector_retriever.build_index(embeddings)
            self.vector_retriever.save_index()
        
        # 初始化BM25索引
        if show_progress:
            self._logger.info("步骤4/6: 构建BM25索引...")
        
        if not force_rebuild and self.bm25_retriever.load_index():
            self._logger.info("已加载缓存的BM25索引")
        else:
            self.bm25_retriever.build_index(texts)
            self.bm25_retriever.save_index()
    
    def _initialize_retrievers(self, show_progress: bool) -> None:
        """初始化检索器。
        
        Args:
            show_progress: 是否显示进度
        """
        if show_progress:
            self._logger.info("步骤5/6: 配置检索器...")
        
        # 设置混合检索器的文档
        chunks = self.dataset_processor.get_processed_chunks()
        self.hybrid_retriever.set_documents(chunks)
        
        self._logger.info("检索器配置完成")
    
    def _validate_system(self, show_progress: bool) -> None:
        """验证系统功能。
        
        Args:
            show_progress: 是否显示进度
        """
        if show_progress:
            self._logger.info("步骤6/6: 验证系统功能...")
        
        # 测试查询
        test_queries = ["测试查询", "什么是人工智能"]
        
        for query in test_queries:
            try:
                result = self.query(query)
                if not result or not result.get('answer'):
                    raise RuntimeError(f"测试查询失败: {query}")
            except Exception as e:
                raise RuntimeError(f"系统验证失败: {e}")
        
        self._logger.info("系统功能验证通过")
    
    def _log_initialization_summary(self) -> None:
        """记录初始化摘要。"""
        chunks = self.dataset_processor.get_processed_chunks()
        vector_stats = self.vector_retriever.get_statistics()
        bm25_stats = self.bm25_retriever.get_statistics()
        
        self._logger.info("初始化摘要:")
        self._logger.info(f"  - 文档块数量: {len(chunks):,}")
        self._logger.info(f"  - 向量索引: {vector_stats.get('num_vectors', 0):,} 个向量")
        self._logger.info(f"  - BM25索引: {bm25_stats.get('num_documents', 0):,} 个文档")
        self._logger.info(f"  - 词汇量: {bm25_stats.get('vocab_size', 0):,}")
        self._logger.info(f"  - 设备: {self.config.device}")
        self._logger.info(f"  - 内存使用: {self._get_memory_usage():.1f} GB")
    
    def query(
        self,
        user_query: str,
        retrieval_strategy: str = "hybrid",
        context_strategy: str = "ranked",
        expand_query: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """处理用户查询。
        
        Args:
            user_query: 用户查询字符串
            retrieval_strategy: 检索策略 ("hybrid", "vector_only", "bm25_only")
            context_strategy: 上下文组织策略 ("ranked", "clustered", "summarized")
            expand_query: 是否进行查询扩展
            include_metadata: 是否包含元数据
            
        Returns:
            包含答案和元信息的结果字典
            
        Raises:
            RuntimeError: 当系统未初始化时抛出
            ValueError: 当查询为空时抛出
        """
        if not self.is_initialized:
            raise RuntimeError("RAG系统未初始化，请先调用 initialize()")
        
        if not user_query.strip():
            raise ValueError("查询不能为空")
        
        self._logger.info(f"处理查询: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'")
        start_time = time.time()
        
        try:
            # 步骤1: 查询预处理和扩展
            processed_query = self._process_user_query(user_query, expand_query)
            
            # 步骤2: 文档检索
            retrieved_docs = self._retrieve_documents(processed_query, retrieval_strategy)
            
            # 步骤3: 上下文组织
            context = self._organize_context(user_query, retrieved_docs, context_strategy, include_metadata)
            
            # 步骤4: 答案生成
            answer = self._generate_answer(user_query, context)
            
            # 步骤5: 构建结果
            result = self._build_query_result(
                user_query, processed_query, answer, retrieved_docs,
                context, start_time
            )
            
            # 更新统计信息
            self._update_query_statistics(time.time() - start_time)
            
            self._logger.info(f"查询处理完成，耗时: {result['processing_time']:.3f} 秒")
            
            return result
            
        except Exception as e:
            self._logger.error(f"查询处理失败: {e}")
            raise
    
    def _process_user_query(self, user_query: str, expand_query: bool) -> str:
        """处理用户查询。
        
        Args:
            user_query: 原始查询
            expand_query: 是否扩展查询
            
        Returns:
            处理后的查询
        """
        # 预处理查询
        processed_query = self.query_processor.preprocess_query(user_query)
        
        # 查询扩展
        if expand_query:
            processed_query = self.query_processor.expand_query(processed_query)
        
        return processed_query
    
    def _retrieve_documents(self, query: str, strategy: str) -> List[Dict[str, Any]]:
        """检索相关文档。
        
        Args:
            query: 查询字符串
            strategy: 检索策略
            
        Returns:
            检索到的文档列表
        """
        return self.hybrid_retriever.search(
            query,
            k=self.config.top_k_retrieval,
            retrieval_strategy=strategy
        )
    
    def _organize_context(
        self,
        original_query: str,
        retrieved_docs: List[Dict[str, Any]],
        strategy: str,
        include_metadata: bool
    ) -> str:
        """组织上下文。
        
        Args:
            original_query: 原始查询
            retrieved_docs: 检索到的文档
            strategy: 组织策略
            include_metadata: 是否包含元数据
            
        Returns:
            组织后的上下文
        """
        return self.context_organizer.organize_context(
            original_query,
            retrieved_docs,
            context_strategy=strategy,
            include_metadata=include_metadata
        )
    
    def _generate_answer(self, query: str, context: str) -> str:
        """生成答案。
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            生成的答案
        """
        try:
            # 使用查询处理器的模型生成答案
            inputs = self.query_processor.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length
            ).to(self.query_processor.device)
            
            # 生成参数
            generation_params = {
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': self.config.temperature,
                'do_sample': self.config.do_sample,
                'pad_token_id': self.query_processor.tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
                'length_penalty': 1.0
            }
            
            # 生成答案
            with torch.no_grad():
                outputs = self.query_processor.model.generate(
                    **inputs,
                    **generation_params
                )
            
            # 解码输出
            generated_text = self.query_processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # 后处理答案
            answer = self._postprocess_answer(generated_text, query)
            
            return answer
            
        except Exception as e:
            self._logger.error(f"答案生成失败: {e}")
            return "抱歉，生成答案时发生错误。请尝试重新提问或联系技术支持。"
    
    def _postprocess_answer(self, generated_text: str, query: str) -> str:
        """后处理生成的答案。
        
        Args:
            generated_text: 生成的文本
            query: 原始查询
            
        Returns:
            后处理后的答案
        """
        if not generated_text:
            return "抱歉，我无法基于提供的信息回答这个问题。"
        
        # 清理答案
        answer = generated_text.strip()
        
        # 移除可能的模板残留
        answer = answer.replace("请根据上述文档内容提供准确、详细的回答：", "")
        answer = answer.replace("根据上述文档内容", "根据相关信息")
        
        # 确保答案完整性
        if len(answer) < 10:
            return "抱歉，生成的答案过短，请尝试重新提问。"
        
        # 截断过长的答案
        max_answer_length = 1000
        if len(answer) > max_answer_length:
            # 尝试在句子边界截断
            sentences = answer.split('。')
            truncated_answer = ""
            for sentence in sentences:
                if len(truncated_answer) + len(sentence) + 1 > max_answer_length:
                    break
                truncated_answer += sentence + "。"
            
            if truncated_answer:
                answer = truncated_answer
            else:
                answer = answer[:max_answer_length] + "..."
        
        return answer
    
    def _build_query_result(
        self,
        original_query: str,
        processed_query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        context: str,
        start_time: float
    ) -> Dict[str, Any]:
        """构建查询结果。
        
        Args:
            original_query: 原始查询
            processed_query: 处理后的查询
            answer: 生成的答案
            retrieved_docs: 检索到的文档
            context: 组织的上下文
            start_time: 开始时间
            
        Returns:
            结果字典
        """
        processing_time = time.time() - start_time
        
        # 基本结果
        result = {
            'query': original_query,
            'processed_query': processed_query,
            'answer': answer,
            'processing_time': processing_time,
            'retrieved_docs_count': len(retrieved_docs),
            'timestamp': time.time()
        }
        
        # 添加检索到的文档（限制数量以节省空间）
        result['retrieved_docs'] = retrieved_docs[:5]  # 只返回前5个
        
        # 添加统计信息
        result['statistics'] = {
            'context_length': len(context),
            'answer_length': len(answer),
            'context_stats': self.context_organizer.get_context_statistics(context)
        }
        
        # 添加性能信息
        result['performance'] = {
            'memory_usage_gb': self._get_memory_usage(),
            'query_count': self._query_count + 1,
            'avg_query_time': (self._total_query_time + processing_time) / (self._query_count + 1)
        }
        
        return result
    
    def _update_query_statistics(self, query_time: float) -> None:
        """更新查询统计信息。
        
        Args:
            query_time: 查询处理时间
        """
        self._query_count += 1
        self._total_query_time += query_time
    
    def batch_query(
        self,
        queries: List[str],
        retrieval_strategy: str = "hybrid",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """批量处理查询。
        
        Args:
            queries: 查询列表
            retrieval_strategy: 检索策略
            show_progress: 是否显示进度
            
        Returns:
            结果列表
        """
        if not self.is_initialized:
            raise RuntimeError("RAG系统未初始化，请先调用 initialize()")
        
        self._logger.info(f"开始批量处理 {len(queries)} 个查询")
        start_time = time.time()
        
        results = []
        
        for i, query in enumerate(queries):
            if show_progress and i % 10 == 0:
                self._logger.info(f"批量查询进度: {i}/{len(queries)}")
            
            try:
                result = self.query(query, retrieval_strategy=retrieval_strategy)
                results.append(result)
            except Exception as e:
                self._logger.error(f"查询 {i} 处理失败: {e}")
                error_result = {
                    'query': query,
                    'answer': f"查询处理失败: {str(e)}",
                    'error': True,
                    'processing_time': 0.0
                }
                results.append(error_result)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if not r.get('error', False))
        
        self._logger.info(f"批量查询完成:")
        self._logger.info(f"  - 成功: {success_count}/{len(queries)}")
        self._logger.info(f"  - 总时间: {total_time:.2f} 秒")
        self._logger.info(f"  - 平均时间: {total_time/len(queries):.3f} 秒/查询")
        
        return results
    
    def explain_query(self, query: str, doc_index: int = 0) -> Dict[str, Any]:
        """解释查询处理过程。
        
        Args:
            query: 查询字符串
            doc_index: 要解释的文档索引
            
        Returns:
            解释信息字典
        """
        if not self.is_initialized:
            raise RuntimeError("RAG系统未初始化，请先调用 initialize()")
        
        explanation = {
            'query': query,
            'query_analysis': self.query_processor.analyze_query(query),
            'retrieval_explanation': self.hybrid_retriever.explain_retrieval(query, doc_index),
            'system_config': {
                'embedding_weight': self.config.embedding_weight,
                'bm25_weight': self.config.bm25_weight,
                'top_k_retrieval': self.config.top_k_retrieval,
                'max_context_length': self.config.max_context_length
            }
        }
        
        return explanation
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息。
        
        Returns:
            系统统计字典
        """
        stats = {
            'initialization': {
                'is_initialized': self.is_initialized,
                'initialization_time': self._initialization_time,
                'config_summary': {
                    'device': self.config.device,
                    'embedding_dim': self.config.embedding_dim,
                    'top_k_retrieval': self.config.top_k_retrieval
                }
            },
            'performance': {
                'total_queries': self._query_count,
                'total_query_time': self._total_query_time,
                'avg_query_time': self._total_query_time / max(self._query_count, 1),
                'memory_usage_gb': self._get_memory_usage()
            },
            'components': {
                'dataset': self.dataset_processor.get_statistics(),
                'vector_retriever': self.vector_retriever.get_statistics(),
                'bm25_retriever': self.bm25_retriever.get_statistics(),
                'hybrid_retriever': self.hybrid_retriever.get_statistics(),
                'query_processor': self.query_processor.get_cache_statistics()
            }
        }
        
        return stats
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（GB）。
        
        Returns:
            内存使用量
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3
            else:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
        except Exception:
            return 0.0
    
    def optimize_performance(self) -> None:
        """优化系统性能。"""
        self._logger.info("开始性能优化...")
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 优化查询处理器缓存
        cache_stats = self.query_processor.get_cache_statistics()
        if cache_stats['cache_size'] > 500:
            self.query_processor.clear_cache()
            self._logger.info("已清理查询处理器缓存")
        
        self._logger.info("性能优化完成")
    
    def save_system_state(self, save_path: str = None) -> None:
        """保存系统状态。
        
        Args:
            save_path: 保存路径前缀
        """
        if not self.is_initialized:
            self._logger.warning("系统未初始化，无法保存状态")
            return
        
        self._logger.info("保存系统状态...")
        
        # 保存各组件状态
        self.vector_retriever.save_index()
        self.bm25_retriever.save_index()
        
        # 保存配置
        config_path = save_path + "_config.yaml" if save_path else "system_config.yaml"
        self.config.save_to_yaml(config_path)
        
        self._logger.info("系统状态保存完成")
    
    def clear_cache(self) -> None:
        """清理所有缓存。"""
        self._logger.info("清理系统缓存...")
        
        # 清理各组件缓存
        self.embedding_processor.clear_cache()
        self.vector_retriever.clear_cache()
        self.bm25_retriever.clear_cache()
        self.hybrid_retriever.clear_cache()
        self.query_processor.clear_cache()
        
        # 重置统计信息
        self._query_count = 0
        self._total_query_time = 0.0
        
        self._logger.info("系统缓存清理完成")


# 导出主要类
__all__ = ['RAGPipeline']