#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular RAG System Usage Example
===============================

演示如何使用模块化的RAG系统。

Author: Claude AI
Date: 2025-10-29
"""

import os
import sys
from typing import List

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAGConfig, load_config
from rag_pipeline import RAGPipeline


def basic_usage_example():
    """基本使用示例。"""
    print("="*60)
    print("模块化RAG系统 - 基本使用示例")
    print("="*60)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    config = RAGConfig(
        embedding_dim=512,          # 使用较小的嵌入维度
        top_k_retrieval=5,          # 检索5个文档
        max_context_length=2048,    # 较短的上下文
        chunk_size=256,             # 较小的文档块
        device="cpu",               # 使用CPU（可改为"cuda"如果有GPU）
        log_level="INFO"
    )
    
    print(f"配置加载完成:")
    print(f"  - 设备: {config.device}")
    print(f"  - 嵌入维度: {config.embedding_dim}")
    print(f"  - 检索数量: {config.top_k_retrieval}")
    
    # 2. 初始化RAG系统
    print("\n2. 初始化RAG系统...")
    print("注意：首次运行需要下载模型和数据集，可能需要较长时间")
    
    rag = RAGPipeline(config)
    
    try:
        rag.initialize(force_rebuild=False)
        print("RAG系统初始化成功！")
        
        # 3. 进行查询
        print("\n3. 测试查询...")
        test_queries = [
            "什么是人工智能？",
            "机器学习的基本原理",
            "深度学习和神经网络的关系"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n查询 {i}: {query}")
            print("-" * 40)
            
            result = rag.query(query)
            
            print(f"答案: {result['answer']}")
            print(f"处理时间: {result['processing_time']:.2f} 秒")
            print(f"检索文档数: {result['retrieved_docs_count']}")
            
            # 显示相关文档
            if result['retrieved_docs']:
                print("\n相关文档:")
                for j, doc in enumerate(result['retrieved_docs'][:2], 1):
                    print(f"  {j}. {doc.get('title', 'N/A')} (分数: {doc.get('retrieval_score', 0):.3f})")
        
        # 4. 系统统计
        print("\n4. 系统统计信息...")
        stats = rag.get_system_statistics()
        print(f"总查询数: {stats['performance']['total_queries']}")
        print(f"平均查询时间: {stats['performance']['avg_query_time']:.3f} 秒")
        print(f"内存使用: {stats['performance']['memory_usage_gb']:.1f} GB")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查系统环境和依赖是否正确安装")


def config_file_example():
    """从配置文件加载的示例。"""
    print("\n" + "="*60)
    print("从配置文件加载示例")
    print("="*60)
    
    # 创建示例配置文件
    config_path = "example_config.yaml"
    
    if not os.path.exists(config_path):
        config = RAGConfig()
        config.save_to_yaml(config_path)
        print(f"已创建示例配置文件: {config_path}")
    
    # 从文件加载配置
    config = load_config(config_path)
    print(f"从配置文件加载完成: {config_path}")
    
    # 使用配置创建RAG系统
    rag = RAGPipeline(config)
    print("RAG系统创建完成")


def custom_components_example():
    """自定义组件示例。"""
    print("\n" + "="*60)
    print("自定义组件示例")
    print("="*60)
    
    # 创建自定义配置
    config = RAGConfig(
        embedding_dim=768,
        bm25_weight=0.4,
        embedding_weight=0.6,
        temperature=0.8,
        do_sample=True
    )
    
    # 创建RAG流水线
    rag = RAGPipeline(config)
    
    # 展示组件访问
    print("可以直接访问各个组件:")
    print(f"  - 数据集处理器: {type(rag.dataset_processor).__name__}")
    print(f"  - 嵌入处理器: {type(rag.embedding_processor).__name__}")
    print(f"  - 向量检索器: {type(rag.vector_retriever).__name__}")
    print(f"  - BM25检索器: {type(rag.bm25_retriever).__name__}")
    print(f"  - 混合检索器: {type(rag.hybrid_retriever).__name__}")
    print(f"  - 查询处理器: {type(rag.query_processor).__name__}")
    print(f"  - 上下文组织器: {type(rag.context_organizer).__name__}")
    
    print("\n各组件可以独立使用和自定义")


def performance_test_example():
    """性能测试示例。"""
    print("\n" + "="*60)
    print("性能测试示例")
    print("="*60)
    
    config = RAGConfig(device="cpu")
    rag = RAGPipeline(config)
    
    # 创建测试查询
    test_queries = [
        "人工智能的定义",
        "机器学习算法分类",
        "深度学习应用领域",
        "自然语言处理技术",
        "计算机视觉发展"
    ]
    
    print(f"准备进行 {len(test_queries)} 个查询的性能测试")
    print("注意：首次运行需要初始化系统，时间较长")
    
    try:
        # 初始化系统
        print("初始化系统...")
        rag.initialize()
        
        # 批量查询测试
        print("开始批量查询测试...")
        results = rag.batch_query(test_queries, show_progress=True)
        
        # 分析结果
        success_count = sum(1 for r in results if not r.get('error', False))
        total_time = sum(r['processing_time'] for r in results if not r.get('error', False))
        avg_time = total_time / success_count if success_count > 0 else 0
        
        print(f"\n性能测试结果:")
        print(f"  - 成功查询: {success_count}/{len(test_queries)}")
        print(f"  - 总时间: {total_time:.2f} 秒")
        print(f"  - 平均时间: {avg_time:.3f} 秒/查询")
        print(f"  - 查询吞吐量: {success_count/total_time:.2f} 查询/秒")
        
    except Exception as e:
        print(f"性能测试失败: {e}")


def interactive_demo():
    """交互式演示。"""
    print("\n" + "="*60)
    print("交互式演示")
    print("="*60)
    
    config = RAGConfig(device="cpu")
    rag = RAGPipeline(config)
    
    try:
        print("正在初始化系统...")
        rag.initialize()
        print("系统初始化完成！")
        
        print("\n欢迎使用模块化RAG系统!")
        print("输入您的问题，输入 'quit' 退出")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("谢谢使用！")
                    break
                
                if not user_input:
                    print("请输入有效的问题")
                    continue
                
                print("正在处理...")
                result = rag.query(user_input)
                
                print(f"\n答案: {result['answer']}")
                print(f"处理时间: {result['processing_time']:.2f} 秒")
                
                # 询问是否显示详细信息
                detail = input("\n显示检索详情？(y/n): ").strip().lower()
                if detail in ['y', 'yes', '是']:
                    print(f"检索到 {result['retrieved_docs_count']} 个相关文档")
                    for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                        print(f"  {i}. {doc.get('title', 'N/A')} (相关性: {doc.get('retrieval_score', 0):.3f})")
                
            except KeyboardInterrupt:
                print("\n\n用户中断，退出程序")
                break
            except Exception as e:
                print(f"处理查询时发生错误: {e}")
                continue
                
    except Exception as e:
        print(f"系统初始化失败: {e}")


def main():
    """主函数。"""
    print("模块化RAG系统演示程序")
    print("=====================")
    
    while True:
        print("\n请选择演示模式:")
        print("1. 基本使用示例")
        print("2. 配置文件示例")
        print("3. 自定义组件示例")
        print("4. 性能测试示例")
        print("5. 交互式演示")
        print("6. 退出")
        
        try:
            choice = input("\n请输入选项 (1-6): ").strip()
            
            if choice == '1':
                basic_usage_example()
            elif choice == '2':
                config_file_example()
            elif choice == '3':
                custom_components_example()
            elif choice == '4':
                performance_test_example()
            elif choice == '5':
                interactive_demo()
            elif choice == '6':
                print("谢谢使用！")
                break
            else:
                print("无效选项，请重新选择")
                
        except KeyboardInterrupt:
            print("\n\n程序被中断，退出")
            break
        except Exception as e:
            print(f"程序执行错误: {e}")
            continue


if __name__ == "__main__":
    main()