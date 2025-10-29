#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Processor Module for RAG System
=====================================

查询处理器模块，负责查询重写、扩展和优化。

Author: Claude AI
Date: 2025-10-29
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import RAGConfig


class QueryProcessor:
    """查询处理器，负责查询重写和扩展。
    
    主要功能：
    1. 查询预处理和清洗
    2. 使用LLM进行查询扩展
    3. 同义词扩展和语义丰富
    4. 查询质量评估和优化
    
    Attributes:
        config: RAG系统配置对象
        model: 语言模型
        tokenizer: 分词器
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化查询处理器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 设备配置
        self.device = torch.device(self.config.device)
        
        # 查询处理缓存
        self._query_cache: Dict[str, str] = {}
        self._cache_enabled = True
        
        self._logger.info(f"查询处理器初始化完成，使用设备: {self.device}")
    
    def load_model(self) -> None:
        """加载语言模型和分词器。
        
        Raises:
            RuntimeError: 当模型加载失败时抛出
        """
        try:
            self._logger.info(f"正在加载语言模型: {self.config.llm_model_name}")
            start_time = time.time()
            
            # 加载分词器
            self._logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            # 加载模型
            self._logger.info("加载语言模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # 移动模型到指定设备（如果没有使用device_map）
            if self.device.type != "cuda" or not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_time = time.time() - start_time
            
            # 显示模型信息
            self._log_model_info()
            
            self._logger.info(f"语言模型加载完成，耗时: {load_time:.2f} 秒")
            
            # 测试模型
            self._test_model()
            
        except Exception as e:
            self._logger.error(f"语言模型加载失败: {e}")
            raise RuntimeError(f"Failed to load language model: {e}")
    
    def _log_model_info(self) -> None:
        """记录模型信息。"""
        if self.model is None:
            return
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self._logger.info(f"模型信息:")
        self._logger.info(f"  - 总参数数: {total_params:,}")
        self._logger.info(f"  - 可训练参数: {trainable_params:,}")
        self._logger.info(f"  - 模型大小: {total_params * 2 / 1024 / 1024:.1f} MB")  # float16
        
        # 显示tokenizer信息
        if self.tokenizer:
            vocab_size = len(self.tokenizer)
            self._logger.info(f"  - 词汇表大小: {vocab_size:,}")
    
    def _test_model(self) -> None:
        """测试模型是否正常工作。"""
        try:
            test_query = "什么是人工智能？"
            expanded_query = self.expand_query(test_query)
            
            self._logger.info(f"模型测试通过")
            self._logger.debug(f"测试查询: '{test_query}' -> '{expanded_query}'")
            
        except Exception as e:
            self._logger.error(f"模型测试失败: {e}")
            raise RuntimeError(f"Model test failed: {e}")
    
    def preprocess_query(self, query: str) -> str:
        """预处理查询字符串。
        
        Args:
            query: 原始查询
            
        Returns:
            预处理后的查询
        """
        if not query:
            return ""
        
        # 基本清理
        processed_query = query.strip()
        
        # 移除多余的空白字符
        processed_query = re.sub(r'\s+', ' ', processed_query)
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        processed_query = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()（），。！？；：]', '', processed_query)
        
        # 限制长度
        max_query_length = 200
        if len(processed_query) > max_query_length:
            processed_query = processed_query[:max_query_length]
            self._logger.warning(f"查询过长，已截断到 {max_query_length} 字符")
        
        return processed_query
    
    def expand_query(
        self,
        query: str,
        expansion_method: str = "llm",
        max_expansion_ratio: float = 2.0
    ) -> str:
        """扩展查询以提高检索效果。
        
        Args:
            query: 原始查询
            expansion_method: 扩展方法 ("llm", "synonym", "none")
            max_expansion_ratio: 最大扩展比例
            
        Returns:
            扩展后的查询
        """
        if not query.strip():
            return query
        
        # 预处理查询
        processed_query = self.preprocess_query(query)
        
        # 检查缓存
        if self._cache_enabled and processed_query in self._query_cache:
            cached_result = self._query_cache[processed_query]
            self._logger.debug(f"使用缓存的查询扩展: '{processed_query}' -> '{cached_result}'")
            return cached_result
        
        # 根据方法进行扩展
        if expansion_method == "llm":
            expanded_query = self._llm_expand_query(processed_query)
        elif expansion_method == "synonym":
            expanded_query = self._synonym_expand_query(processed_query)
        else:  # none
            expanded_query = processed_query
        
        # 检查扩展比例
        if len(expanded_query) > len(processed_query) * max_expansion_ratio:
            self._logger.warning(f"查询扩展过度，限制长度")
            expanded_query = expanded_query[:int(len(processed_query) * max_expansion_ratio)]
        
        # 更新缓存
        if self._cache_enabled:
            self._query_cache[processed_query] = expanded_query
            
            # 限制缓存大小
            if len(self._query_cache) > 1000:
                # 移除最早的条目
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
        
        if expanded_query != processed_query:
            self._logger.debug(f"查询已扩展: '{processed_query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def _llm_expand_query(self, query: str) -> str:
        """使用LLM扩展查询。
        
        Args:
            query: 原始查询
            
        Returns:
            LLM扩展后的查询
        """
        if self.model is None or self.tokenizer is None:
            self._logger.warning("模型未加载，返回原始查询")
            return query
        
        # 构建查询扩展提示
        prompt = self._build_expansion_prompt(query)
        
        try:
            # 生成扩展查询
            expanded_query = self._generate_expansion(prompt, query)
            
            # 验证扩展结果
            if self._is_valid_expansion(query, expanded_query):
                return expanded_query
            else:
                self._logger.warning(f"LLM扩展结果无效，返回原始查询")
                return query
                
        except Exception as e:
            self._logger.error(f"LLM查询扩展失败: {e}")
            return query
    
    def _build_expansion_prompt(self, query: str) -> str:
        """构建查询扩展的提示词。
        
        Args:
            query: 原始查询
            
        Returns:
            提示词字符串
        """
        prompt = f"""请帮助扩展以下查询，添加相关的同义词、概念和表达方式，以提高搜索效果。
要求：
1. 保持原始查询的核心意图
2. 添加相关的同义词和近义词
3. 包含相关的概念和术语
4. 保持简洁，不要过度扩展
5. 只输出扩展后的查询，不要包含其他内容

原始查询: {query}
扩展查询:"""
        
        return prompt
    
    def _generate_expansion(self, prompt: str, original_query: str) -> str:
        """生成查询扩展。
        
        Args:
            prompt: 提示词
            original_query: 原始查询
            
        Returns:
            生成的扩展查询
        """
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(self.device)
        
        # 生成参数
        generation_params = {
            'max_new_tokens': min(100, len(original_query) * 3),
            'temperature': self.config.temperature,
            'do_sample': self.config.do_sample,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': 1.1,
            'length_penalty': 0.8
        }
        
        # 生成扩展查询
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 后处理
        expanded_query = self._postprocess_generated_text(generated_text, original_query)
        
        return expanded_query
    
    def _postprocess_generated_text(self, generated_text: str, original_query: str) -> str:
        """后处理生成的文本。
        
        Args:
            generated_text: 生成的文本
            original_query: 原始查询
            
        Returns:
            后处理后的扩展查询
        """
        if not generated_text:
            return original_query
        
        # 移除换行符和多余空白
        cleaned_text = re.sub(r'\n+', ' ', generated_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 移除可能的提示词残留
        patterns_to_remove = [
            r'^扩展查询[:：]\s*',
            r'^查询[:：]\s*',
            r'^答案[:：]\s*',
            r'^结果[:：]\s*'
        ]
        
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 取第一句或第一行
        lines = cleaned_text.split('\n')
        if lines:
            cleaned_text = lines[0].strip()
        
        # 移除可能的结束标记
        end_markers = ['。', '.', '!', '！', '?', '？']
        for marker in end_markers:
            if cleaned_text.endswith(marker):
                cleaned_text = cleaned_text[:-1].strip()
                break
        
        # 如果结果太短或太相似，返回原始查询
        if len(cleaned_text) < len(original_query) * 0.8:
            return original_query
        
        return cleaned_text
    
    def _synonym_expand_query(self, query: str) -> str:
        """使用同义词扩展查询（简单版本）。
        
        Args:
            query: 原始查询
            
        Returns:
            同义词扩展后的查询
        """
        # 简单的同义词映射
        synonym_map = {
            # 中文同义词
            '人工智能': '人工智能 AI 机器智能',
            '机器学习': '机器学习 ML 机器学习算法',
            '深度学习': '深度学习 DL 神经网络 深度神经网络',
            '自然语言处理': '自然语言处理 NLP 语言处理',
            '计算机': '计算机 电脑 PC',
            '软件': '软件 程序 应用程序',
            '数据': '数据 信息 资料',
            '算法': '算法 方法 技术',
            
            # 英文同义词
            'artificial intelligence': 'artificial intelligence AI machine intelligence',
            'machine learning': 'machine learning ML',
            'deep learning': 'deep learning DL neural networks',
            'computer': 'computer PC device',
            'software': 'software program application',
            'data': 'data information',
            'algorithm': 'algorithm method technique'
        }
        
        expanded_query = query
        
        for term, expansion in synonym_map.items():
            if term.lower() in query.lower():
                expanded_query = expanded_query + ' ' + expansion
                break  # 只扩展一个主要术语
        
        return expanded_query.strip()
    
    def _is_valid_expansion(self, original_query: str, expanded_query: str) -> bool:
        """验证扩展查询是否有效。
        
        Args:
            original_query: 原始查询
            expanded_query: 扩展查询
            
        Returns:
            是否有效
        """
        if not expanded_query or not expanded_query.strip():
            return False
        
        # 检查是否包含原始查询的关键信息
        original_words = set(original_query.lower().split())
        expanded_words = set(expanded_query.lower().split())
        
        # 至少保留50%的原始词汇
        overlap_ratio = len(original_words & expanded_words) / len(original_words)
        if overlap_ratio < 0.5:
            return False
        
        # 长度检查
        if len(expanded_query) > len(original_query) * 3:
            return False
        
        # 质量检查（避免重复词汇过多）
        words = expanded_query.split()
        unique_words = set(words)
        if len(words) > 5 and len(unique_words) / len(words) < 0.7:
            return False
        
        return True
    
    def batch_expand_queries(
        self,
        queries: List[str],
        expansion_method: str = "llm"
    ) -> List[str]:
        """批量扩展查询。
        
        Args:
            queries: 查询列表
            expansion_method: 扩展方法
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = []
        
        self._logger.info(f"批量扩展 {len(queries)} 个查询")
        start_time = time.time()
        
        for i, query in enumerate(queries):
            if i % 50 == 0 and i > 0:
                self._logger.debug(f"批量扩展进度: {i}/{len(queries)}")
            
            try:
                expanded_query = self.expand_query(query, expansion_method)
                expanded_queries.append(expanded_query)
            except Exception as e:
                self._logger.error(f"查询 {i} 扩展失败: {e}")
                expanded_queries.append(query)  # 使用原始查询
        
        total_time = time.time() - start_time
        self._logger.info(f"批量查询扩展完成，耗时 {total_time:.2f} 秒")
        
        return expanded_queries
    
    def analyze_query(self, query: str) -> Dict[str, any]:
        """分析查询特征。
        
        Args:
            query: 查询字符串
            
        Returns:
            查询分析结果
        """
        analysis = {
            'original_query': query,
            'processed_query': self.preprocess_query(query),
            'length': len(query),
            'word_count': len(query.split()),
            'has_chinese': bool(re.search(r'[\u4e00-\u9fff]', query)),
            'has_english': bool(re.search(r'[a-zA-Z]', query)),
            'has_numbers': bool(re.search(r'\d', query)),
            'question_words': [],
            'query_type': 'unknown'
        }
        
        # 检测疑问词
        question_words_cn = ['什么', '怎么', '为什么', '哪个', '哪些', '何时', '何地', '如何']
        question_words_en = ['what', 'how', 'why', 'which', 'when', 'where', 'who']
        
        query_lower = query.lower()
        for word in question_words_cn + question_words_en:
            if word in query_lower:
                analysis['question_words'].append(word)
        
        # 判断查询类型
        if analysis['question_words']:
            analysis['query_type'] = 'question'
        elif '？' in query or '?' in query:
            analysis['query_type'] = 'question'
        elif len(query.split()) <= 3:
            analysis['query_type'] = 'keyword'
        else:
            analysis['query_type'] = 'description'
        
        return analysis
    
    def get_cache_statistics(self) -> Dict[str, any]:
        """获取缓存统计信息。
        
        Returns:
            缓存统计字典
        """
        return {
            'cache_enabled': self._cache_enabled,
            'cache_size': len(self._query_cache),
            'cache_hit_ratio': self._calculate_cache_hit_ratio()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """计算缓存命中率（简化版本）。
        
        Returns:
            缓存命中率
        """
        # 这里是简化实现，实际应用中可以记录更详细的统计
        return 0.0 if not self._query_cache else len(self._query_cache) / 1000
    
    def clear_cache(self) -> None:
        """清理缓存。"""
        self._query_cache.clear()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._logger.info("查询处理器缓存已清理")
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """设置缓存是否启用。
        
        Args:
            enabled: 是否启用缓存
        """
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        
        self._logger.info(f"查询缓存{'启用' if enabled else '禁用'}")


# 导出主要类
__all__ = ['QueryProcessor']