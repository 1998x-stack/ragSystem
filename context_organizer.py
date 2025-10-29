#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Organizer Module for RAG System
=======================================

上下文组织器模块，负责整理检索到的文档为LLM输入格式。

Author: Claude AI
Date: 2025-10-29
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config import RAGConfig


class ContextOrganizer:
    """上下文组织器，负责整理检索到的文档为模型输入。
    
    主要功能：
    1. 文档内容组织和格式化
    2. 上下文长度控制和截断
    3. 相关性排序和筛选
    4. 元信息标注和结构化
    
    Attributes:
        config: RAG系统配置对象
        _logger: 日志记录器
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """初始化上下文组织器。
        
        Args:
            config: RAG系统配置对象
        """
        self.config = config
        self._logger = logger.bind(name=self.__class__.__name__)
        
        # 上下文组织参数
        self.max_context_length = config.max_context_length
        self.context_template = self._load_context_template()
        
        self._logger.info("上下文组织器初始化完成")
        self._logger.info(f"最大上下文长度: {self.max_context_length}")
    
    def _load_context_template(self) -> str:
        """加载上下文模板。
        
        Returns:
            上下文模板字符串
        """
        # 默认的上下文模板
        template = """基于以下检索到的文档内容，请回答用户的问题。如果文档中没有相关信息，请明确说明。

检索到的相关文档：
{context_content}

用户问题：{user_query}

请根据上述文档内容提供准确、详细的回答："""
        
        return template
    
    def organize_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        context_strategy: str = "ranked",
        include_metadata: bool = True
    ) -> str:
        """组织上下文信息。
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档列表
            context_strategy: 上下文组织策略 ("ranked", "clustered", "summarized")
            include_metadata: 是否包含元数据
            
        Returns:
            组织好的上下文字符串
        """
        if not retrieved_docs:
            return self._create_empty_context(query)
        
        self._logger.info(f"组织上下文，文档数量: {len(retrieved_docs)}")
        start_time = time.time()
        
        try:
            # 预处理文档
            processed_docs = self._preprocess_documents(retrieved_docs)
            
            # 根据策略组织上下文
            if context_strategy == "ranked":
                context_content = self._organize_ranked_context(processed_docs, include_metadata)
            elif context_strategy == "clustered":
                context_content = self._organize_clustered_context(processed_docs, include_metadata)
            elif context_strategy == "summarized":
                context_content = self._organize_summarized_context(processed_docs, include_metadata)
            else:
                context_content = self._organize_ranked_context(processed_docs, include_metadata)
            
            # 应用长度限制
            context_content = self._apply_length_limit(context_content, query)
            
            # 生成最终上下文
            final_context = self.context_template.format(
                context_content=context_content,
                user_query=query
            )
            
            organization_time = time.time() - start_time
            
            self._logger.info(f"上下文组织完成:")
            self._logger.info(f"  - 最终长度: {len(final_context)} 字符")
            self._logger.info(f"  - 使用文档: {len(self._count_used_documents(context_content))} 个")
            self._logger.info(f"  - 组织时间: {organization_time:.3f} 秒")
            
            return final_context
            
        except Exception as e:
            self._logger.error(f"上下文组织失败: {e}")
            return self._create_fallback_context(query, retrieved_docs)
    
    def _preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预处理文档。
        
        Args:
            documents: 原始文档列表
            
        Returns:
            预处理后的文档列表
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._preprocess_single_document(doc, i)
                if processed_doc:
                    processed_docs.append(processed_doc)
            except Exception as e:
                self._logger.warning(f"文档 {i} 预处理失败: {e}")
                continue
        
        return processed_docs
    
    def _preprocess_single_document(self, doc: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """预处理单个文档。
        
        Args:
            doc: 原始文档
            index: 文档索引
            
        Returns:
            预处理后的文档，如果无效则返回None
        """
        # 提取和清理文本内容
        text_content = doc.get('text', '').strip()
        if not text_content:
            return None
        
        # 清理文本
        cleaned_text = self._clean_text(text_content)
        if len(cleaned_text) < 10:  # 过滤过短的文档
            return None
        
        # 构建处理后的文档
        processed_doc = {
            'index': index,
            'id': doc.get('id', f'doc_{index}'),
            'title': doc.get('title', ''),
            'text': cleaned_text,
            'retrieval_score': doc.get('retrieval_score', 0.0),
            'vector_score': doc.get('vector_score', 0.0),
            'bm25_score': doc.get('bm25_score', 0.0),
            'source_info': self._extract_source_info(doc),
            'text_length': len(cleaned_text)
        }
        
        return processed_doc
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容。
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        cleaned = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符和控制字符
        cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        # 标准化引号
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        # 移除多余的标点符号
        cleaned = re.sub(r'([.!?]){2,}', r'\1', cleaned)
        
        return cleaned.strip()
    
    def _extract_source_info(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """提取文档源信息。
        
        Args:
            doc: 文档字典
            
        Returns:
            源信息字典
        """
        source_info = {}
        
        # 提取可能的源信息字段
        source_fields = ['source_idx', 'chunk_idx', 'original_length', 'retrieval_method']
        for field in source_fields:
            if field in doc:
                source_info[field] = doc[field]
        
        return source_info
    
    def _organize_ranked_context(
        self,
        documents: List[Dict[str, Any]],
        include_metadata: bool
    ) -> str:
        """按相关性排序组织上下文。
        
        Args:
            documents: 预处理后的文档列表
            include_metadata: 是否包含元数据
            
        Returns:
            组织后的上下文内容
        """
        # 按检索分数排序
        sorted_docs = sorted(
            documents,
            key=lambda x: x['retrieval_score'],
            reverse=True
        )
        
        context_parts = []
        current_length = 0
        used_docs = 0
        
        for doc in sorted_docs:
            # 构建文档内容
            doc_content = self._format_document_content(doc, used_docs + 1, include_metadata)
            
            # 检查长度限制
            if current_length + len(doc_content) > self._get_available_context_length():
                # 尝试截断当前文档
                remaining_length = self._get_available_context_length() - current_length
                if remaining_length > 200:  # 至少保留200字符
                    truncated_content = self._truncate_document_content(
                        doc_content, remaining_length
                    )
                    context_parts.append(truncated_content)
                break
            
            context_parts.append(doc_content)
            current_length += len(doc_content)
            used_docs += 1
        
        if used_docs < len(sorted_docs):
            self._logger.info(f"由于长度限制，使用了 {used_docs}/{len(sorted_docs)} 个文档")
        
        return '\n\n'.join(context_parts)
    
    def _organize_clustered_context(
        self,
        documents: List[Dict[str, Any]],
        include_metadata: bool
    ) -> str:
        """按主题聚类组织上下文。
        
        Args:
            documents: 预处理后的文档列表
            include_metadata: 是否包含元数据
            
        Returns:
            组织后的上下文内容
        """
        # 简化的聚类：按标题相似性分组
        clusters = self._simple_cluster_documents(documents)
        
        context_parts = []
        current_length = 0
        
        for cluster_idx, cluster_docs in enumerate(clusters):
            if current_length >= self._get_available_context_length():
                break
            
            # 为每个聚类添加标题
            cluster_title = f"相关主题 {cluster_idx + 1}:"
            context_parts.append(cluster_title)
            current_length += len(cluster_title)
            
            # 添加聚类中的文档
            for doc_idx, doc in enumerate(cluster_docs):
                doc_content = self._format_document_content(doc, doc_idx + 1, include_metadata)
                
                if current_length + len(doc_content) > self._get_available_context_length():
                    break
                
                context_parts.append(doc_content)
                current_length += len(doc_content)
        
        return '\n\n'.join(context_parts)
    
    def _organize_summarized_context(
        self,
        documents: List[Dict[str, Any]],
        include_metadata: bool
    ) -> str:
        """组织摘要式上下文。
        
        Args:
            documents: 预处理后的文档列表
            include_metadata: 是否包含元数据
            
        Returns:
            组织后的上下文内容
        """
        # 选择最相关的文档进行详细展示
        top_docs = sorted(
            documents,
            key=lambda x: x['retrieval_score'],
            reverse=True
        )[:3]  # 只详细展示前3个
        
        # 其他文档进行摘要
        other_docs = documents[3:] if len(documents) > 3 else []
        
        context_parts = []
        current_length = 0
        
        # 详细展示顶部文档
        if top_docs:
            context_parts.append("最相关的文档内容：")
            current_length += len(context_parts[-1])
            
            for i, doc in enumerate(top_docs):
                doc_content = self._format_document_content(doc, i + 1, include_metadata)
                
                if current_length + len(doc_content) > self._get_available_context_length():
                    break
                
                context_parts.append(doc_content)
                current_length += len(doc_content)
        
        # 摘要展示其他文档
        if other_docs and current_length < self._get_available_context_length():
            summary_content = self._create_documents_summary(other_docs)
            
            if current_length + len(summary_content) <= self._get_available_context_length():
                context_parts.append("\n其他相关信息摘要：")
                context_parts.append(summary_content)
        
        return '\n\n'.join(context_parts)
    
    def _simple_cluster_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """简单的文档聚类。
        
        Args:
            documents: 文档列表
            
        Returns:
            聚类后的文档组列表
        """
        # 简化实现：按标题关键词分组
        clusters = []
        used_docs = set()
        
        for doc in documents:
            if doc['index'] in used_docs:
                continue
            
            # 创建新聚类
            cluster = [doc]
            used_docs.add(doc['index'])
            
            # 查找相似文档
            doc_keywords = self._extract_keywords(doc['title'] + ' ' + doc['text'][:200])
            
            for other_doc in documents:
                if other_doc['index'] in used_docs:
                    continue
                
                other_keywords = self._extract_keywords(
                    other_doc['title'] + ' ' + other_doc['text'][:200]
                )
                
                # 计算关键词重叠
                overlap = len(doc_keywords & other_keywords)
                if overlap >= 2:  # 至少2个共同关键词
                    cluster.append(other_doc)
                    used_docs.add(other_doc['index'])
            
            clusters.append(cluster)
            
            # 限制聚类数量
            if len(clusters) >= 5:
                break
        
        return clusters
    
    def _extract_keywords(self, text: str) -> set:
        """提取文本关键词。
        
        Args:
            text: 输入文本
            
        Returns:
            关键词集合
        """
        # 简单的关键词提取
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 过滤停用词
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            '的', '了', '是', '在', '有', '和', '就', '不', '与', '也', '都', '要', '可以'
        }
        
        keywords = {word for word in words if word not in stopwords and len(word) > 2}
        
        return keywords
    
    def _format_document_content(
        self,
        doc: Dict[str, Any],
        doc_number: int,
        include_metadata: bool
    ) -> str:
        """格式化文档内容。
        
        Args:
            doc: 文档字典
            doc_number: 文档编号
            include_metadata: 是否包含元数据
            
        Returns:
            格式化后的文档内容
        """
        content_parts = []
        
        # 文档标题
        title = doc.get('title', '').strip()
        if title:
            header = f"文档 {doc_number}: {title}"
        else:
            header = f"文档 {doc_number}"
        
        content_parts.append(header)
        
        # 元数据信息
        if include_metadata:
            metadata_parts = []
            
            # 相关性分数
            retrieval_score = doc.get('retrieval_score', 0.0)
            if retrieval_score > 0:
                metadata_parts.append(f"相关性: {retrieval_score:.3f}")
            
            # 检索方法
            if 'retrieval_method' in doc.get('source_info', {}):
                method = doc['source_info']['retrieval_method']
                metadata_parts.append(f"检索方法: {method}")
            
            if metadata_parts:
                content_parts.append(f"({', '.join(metadata_parts)})")
        
        # 文档内容
        content_parts.append(doc['text'])
        
        # 分隔线
        content_parts.append("-" * 40)
        
        return '\n'.join(content_parts)
    
    def _create_documents_summary(self, documents: List[Dict[str, Any]]) -> str:
        """创建文档摘要。
        
        Args:
            documents: 文档列表
            
        Returns:
            文档摘要
        """
        if not documents:
            return ""
        
        summary_parts = []
        
        for doc in documents[:5]:  # 最多摘要5个文档
            title = doc.get('title', '').strip()
            text_snippet = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
            
            if title:
                summary_parts.append(f"• {title}: {text_snippet}")
            else:
                summary_parts.append(f"• {text_snippet}")
        
        if len(documents) > 5:
            summary_parts.append(f"...以及其他 {len(documents) - 5} 个相关文档")
        
        return '\n'.join(summary_parts)
    
    def _apply_length_limit(self, context_content: str, query: str) -> str:
        """应用长度限制。
        
        Args:
            context_content: 上下文内容
            query: 用户查询
            
        Returns:
            长度限制后的上下文内容
        """
        available_length = self._get_available_context_length()
        
        if len(context_content) <= available_length:
            return context_content
        
        self._logger.warning(f"上下文超长，从 {len(context_content)} 截断到 {available_length}")
        
        # 智能截断：尽量保持文档完整性
        return self._intelligent_truncate(context_content, available_length)
    
    def _get_available_context_length(self) -> int:
        """获取可用的上下文长度。
        
        Returns:
            可用长度
        """
        # 为模板和查询预留空间
        template_overhead = len(self.context_template) - len("{context_content}") - len("{user_query}")
        query_space = 200  # 为查询预留的空间
        response_space = 500  # 为生成回答预留的空间
        
        available_length = self.max_context_length - template_overhead - query_space - response_space
        
        return max(available_length, 500)  # 最少保证500字符
    
    def _intelligent_truncate(self, content: str, max_length: int) -> str:
        """智能截断上下文。
        
        Args:
            content: 原始内容
            max_length: 最大长度
            
        Returns:
            截断后的内容
        """
        if len(content) <= max_length:
            return content
        
        # 按文档分界线分割
        parts = content.split('-' * 40)
        
        if len(parts) <= 1:
            # 没有文档分界线，直接截断
            return content[:max_length] + "...\n[内容因长度限制被截断]"
        
        # 逐个添加文档，直到接近长度限制
        result_parts = []
        current_length = 0
        
        for part in parts[:-1]:  # 最后一个通常是空的
            part_length = len(part) + len('-' * 40)
            
            if current_length + part_length > max_length:
                break
            
            result_parts.append(part)
            current_length += part_length
        
        if result_parts:
            result = ('-' * 40).join(result_parts)
            if len(parts) - 1 > len(result_parts):
                result += f"\n\n[还有 {len(parts) - 1 - len(result_parts)} 个文档因长度限制未显示]"
            return result
        else:
            # 如果连第一个文档都太长，截断第一个文档
            first_part = parts[0][:max_length - 100]
            return first_part + "...\n[内容因长度限制被截断]"
    
    def _truncate_document_content(self, doc_content: str, max_length: int) -> str:
        """截断单个文档内容。
        
        Args:
            doc_content: 文档内容
            max_length: 最大长度
            
        Returns:
            截断后的内容
        """
        if len(doc_content) <= max_length:
            return doc_content
        
        # 尝试在句子边界截断
        sentences = doc_content.split('。')
        if len(sentences) > 1:
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) + 1 > max_length - 20:
                    break
                truncated += sentence + "。"
            
            if truncated:
                return truncated + "\n[文档内容因长度限制被截断]"
        
        # 直接截断
        return doc_content[:max_length - 20] + "...\n[文档内容因长度限制被截断]"
    
    def _count_used_documents(self, context_content: str) -> int:
        """计算上下文中使用的文档数量。
        
        Args:
            context_content: 上下文内容
            
        Returns:
            文档数量
        """
        # 通过分界线计算文档数量
        return context_content.count('-' * 40)
    
    def _create_empty_context(self, query: str) -> str:
        """创建空上下文。
        
        Args:
            query: 用户查询
            
        Returns:
            空上下文字符串
        """
        empty_content = "抱歉，没有找到与您的问题相关的文档内容。"
        
        return self.context_template.format(
            context_content=empty_content,
            user_query=query
        )
    
    def _create_fallback_context(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """创建后备上下文。
        
        Args:
            query: 用户查询
            documents: 文档列表
            
        Returns:
            后备上下文字符串
        """
        # 简单地连接前几个文档的文本
        fallback_content = ""
        
        for i, doc in enumerate(documents[:3]):
            text = doc.get('text', '').strip()
            if text:
                fallback_content += f"文档 {i + 1}: {text[:500]}...\n\n"
        
        if not fallback_content:
            fallback_content = "文档内容获取失败。"
        
        return self.context_template.format(
            context_content=fallback_content,
            user_query=query
        )
    
    def get_context_statistics(self, context: str) -> Dict[str, Any]:
        """获取上下文统计信息。
        
        Args:
            context: 上下文字符串
            
        Returns:
            统计信息字典
        """
        return {
            'total_length': len(context),
            'word_count': len(context.split()),
            'document_count': self._count_used_documents(context),
            'utilization_ratio': len(context) / self.max_context_length,
            'has_truncation': '[截断]' in context or '[未显示]' in context
        }
    
    def set_context_template(self, template: str) -> None:
        """设置上下文模板。
        
        Args:
            template: 新的模板字符串，应包含{context_content}和{user_query}占位符
        """
        if '{context_content}' not in template or '{user_query}' not in template:
            raise ValueError("模板必须包含 {context_content} 和 {user_query} 占位符")
        
        self.context_template = template
        self._logger.info("上下文模板已更新")


# 导出主要类
__all__ = ['ContextOrganizer']