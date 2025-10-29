#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Import and Basic Functionality Test
==========================================

测试所有模块的导入和基本功能。

Author: Claude AI
Date: 2025-10-29
"""

import sys
import traceback
from typing import Dict, Any


def test_imports() -> Dict[str, bool]:
    """测试所有模块的导入。
    
    Returns:
        导入测试结果字典
    """
    print("="*60)
    print("模块导入测试")
    print("="*60)
    
    import_results = {}
    
    # 测试核心模块导入
    modules_to_test = [
        ('config', 'RAGConfig, LoggerManager, load_config'),
        ('dataset_processor', 'DatasetProcessor'),
        ('embedding_processor', 'EmbeddingProcessor'),
        ('vector_retriever', 'VectorRetriever'),
        ('bm25_retriever', 'BM25Retriever'),
        ('hybrid_retriever', 'HybridRetriever'),
        ('query_processor', 'QueryProcessor'),
        ('context_organizer', 'ContextOrganizer'),
        ('rag_pipeline', 'RAGPipeline')
    ]
    
    for module_name, classes in modules_to_test:
        try:
            exec(f"from {module_name} import {classes}")
            import_results[module_name] = True
            print(f"✓ {module_name}: 导入成功")
        except Exception as e:
            import_results[module_name] = False
            print(f"✗ {module_name}: 导入失败 - {e}")
    
    # 总结
    success_count = sum(import_results.values())
    total_count = len(import_results)
    print(f"\n导入测试完成: {success_count}/{total_count} 个模块成功")
    
    return import_results


def test_config_module() -> bool:
    """测试配置模块。
    
    Returns:
        是否测试成功
    """
    print("\n" + "="*60)
    print("配置模块测试")
    print("="*60)
    
    try:
        from config import RAGConfig, LoggerManager
        
        # 测试默认配置创建
        config = RAGConfig()
        print(f"✓ 默认配置创建成功")
        print(f"  - 设备: {config.device}")
        print(f"  - 嵌入维度: {config.embedding_dim}")
        print(f"  - 检索数量: {config.top_k_retrieval}")
        
        # 测试自定义配置
        custom_config = RAGConfig(
            embedding_dim=1024,
            top_k_retrieval=5
        )
        print(f"✓ 自定义配置创建成功")
        
        # 测试配置转换
        config_dict = config.to_dict()
        print(f"✓ 配置转字典成功，包含 {len(config_dict)} 个参数")
        
        # 测试日志管理器
        LoggerManager.setup_logger(config)
        print(f"✓ 日志系统设置成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置模块测试失败: {e}")
        traceback.print_exc()
        return False


def test_basic_components() -> bool:
    """测试基本组件创建。
    
    Returns:
        是否测试成功
    """
    print("\n" + "="*60)
    print("基本组件测试")
    print("="*60)
    
    try:
        from config import RAGConfig
        from dataset_processor import DatasetProcessor
        from embedding_processor import EmbeddingProcessor
        from vector_retriever import VectorRetriever
        from bm25_retriever import BM25Retriever
        from context_organizer import ContextOrganizer
        
        # 创建测试配置
        config = RAGConfig(
            device="cpu",
            embedding_dim=128,  # 小维度用于测试
            chunk_size=100,
            max_context_length=500
        )
        
        # 测试各组件创建
        components = [
            ("数据集处理器", DatasetProcessor),
            ("嵌入处理器", EmbeddingProcessor),
            ("向量检索器", VectorRetriever),
            ("BM25检索器", BM25Retriever),
            ("上下文组织器", ContextOrganizer)
        ]
        
        created_components = {}
        
        for name, component_class in components:
            try:
                component = component_class(config)
                created_components[name] = component
                print(f"✓ {name}: 创建成功")
            except Exception as e:
                print(f"✗ {name}: 创建失败 - {e}")
                return False
        
        # 测试上下文组织器的基本功能
        context_organizer = created_components["上下文组织器"]
        test_docs = [
            {"id": "1", "title": "测试文档1", "text": "这是第一个测试文档", "retrieval_score": 0.9},
            {"id": "2", "title": "测试文档2", "text": "这是第二个测试文档", "retrieval_score": 0.8}
        ]
        
        context = context_organizer.organize_context("测试查询", test_docs)
        print(f"✓ 上下文组织测试成功，长度: {len(context)} 字符")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本组件测试失败: {e}")
        traceback.print_exc()
        return False


def test_dependencies() -> Dict[str, str]:
    """测试依赖包。
    
    Returns:
        依赖包状态字典
    """
    print("\n" + "="*60)
    print("依赖包测试")
    print("="*60)
    
    dependencies = {
        'torch': None,
        'transformers': None,
        'datasets': None,
        'numpy': None,
        'faiss': None,
        'rank_bm25': None,
        'loguru': None,
        'yaml': None
    }
    
    # 测试 PyTorch
    try:
        import torch
        dependencies['torch'] = torch.__version__
        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - CUDA 可用: {torch.cuda.device_count()} 个设备")
        else:
            print(f"  - CUDA 不可用，将使用 CPU")
    except ImportError:
        dependencies['torch'] = "未安装"
        print(f"✗ PyTorch: 未安装")
    
    # 测试 Transformers
    try:
        import transformers
        dependencies['transformers'] = transformers.__version__
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        dependencies['transformers'] = "未安装"
        print(f"✗ Transformers: 未安装")
    
    # 测试 Datasets
    try:
        import datasets
        dependencies['datasets'] = datasets.__version__
        print(f"✓ Datasets: {datasets.__version__}")
    except ImportError:
        dependencies['datasets'] = "未安装"
        print(f"✗ Datasets: 未安装")
    
    # 测试 NumPy
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        dependencies['numpy'] = "未安装"
        print(f"✗ NumPy: 未安装")
    
    # 测试 FAISS
    try:
        import faiss
        dependencies['faiss'] = "已安装"
        print(f"✓ FAISS: 已安装")
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            print(f"  - GPU 支持: {faiss.get_num_gpus()} 个GPU")
        else:
            print(f"  - 仅 CPU 支持")
    except ImportError:
        dependencies['faiss'] = "未安装"
        print(f"✗ FAISS: 未安装")
    
    # 测试 BM25
    try:
        from rank_bm25 import BM25Okapi
        dependencies['rank_bm25'] = "已安装"
        print(f"✓ Rank-BM25: 已安装")
    except ImportError:
        dependencies['rank_bm25'] = "未安装"
        print(f"✗ Rank-BM25: 未安装")
    
    # 测试 Loguru
    try:
        import loguru
        dependencies['loguru'] = loguru.__version__
        print(f"✓ Loguru: {loguru.__version__}")
    except ImportError:
        dependencies['loguru'] = "未安装"
        print(f"✗ Loguru: 未安装")
    
    # 测试 PyYAML
    try:
        import yaml
        dependencies['yaml'] = "已安装"
        print(f"✓ PyYAML: 已安装")
    except ImportError:
        dependencies['yaml'] = "未安装"
        print(f"✗ PyYAML: 未安装")
    
    return dependencies


def test_minimal_pipeline() -> bool:
    """测试最小化流水线创建。
    
    Returns:
        是否测试成功
    """
    print("\n" + "="*60)
    print("最小化流水线测试")
    print("="*60)
    
    try:
        from config import RAGConfig
        from rag_pipeline import RAGPipeline
        
        # 创建最小配置
        config = RAGConfig(
            device="cpu",
            embedding_dim=64,  # 极小维度
            chunk_size=50,
            max_context_length=200,
            top_k_retrieval=2
        )
        
        # 创建流水线
        rag = RAGPipeline(config)
        print(f"✓ RAG流水线创建成功")
        
        # 检查组件
        components = [
            "dataset_processor",
            "embedding_processor", 
            "vector_retriever",
            "bm25_retriever",
            "hybrid_retriever",
            "query_processor",
            "context_organizer"
        ]
        
        for component_name in components:
            if hasattr(rag, component_name):
                print(f"✓ 组件 {component_name}: 存在")
            else:
                print(f"✗ 组件 {component_name}: 缺失")
                return False
        
        print(f"✓ 所有组件检查通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 最小化流水线测试失败: {e}")
        traceback.print_exc()
        return False


def generate_test_report(
    import_results: Dict[str, bool],
    config_test: bool,
    components_test: bool,
    dependencies: Dict[str, str],
    pipeline_test: bool
) -> None:
    """生成测试报告。
    
    Args:
        import_results: 导入测试结果
        config_test: 配置测试结果
        components_test: 组件测试结果
        dependencies: 依赖包状态
        pipeline_test: 流水线测试结果
    """
    print("\n" + "="*60)
    print("测试报告")
    print("="*60)
    
    # 导入测试总结
    import_success = sum(import_results.values())
    import_total = len(import_results)
    print(f"模块导入: {import_success}/{import_total} 成功")
    
    # 功能测试总结
    tests = [
        ("配置模块", config_test),
        ("基本组件", components_test),
        ("流水线创建", pipeline_test)
    ]
    
    passed_tests = sum(1 for _, result in tests if result)
    total_tests = len(tests)
    print(f"功能测试: {passed_tests}/{total_tests} 通过")
    
    # 依赖包总结
    installed_deps = sum(1 for dep in dependencies.values() if dep and dep != "未安装")
    total_deps = len(dependencies)
    print(f"依赖包: {installed_deps}/{total_deps} 已安装")
    
    # 整体状态
    overall_success = (
        import_success == import_total and
        passed_tests == total_tests and
        installed_deps >= total_deps * 0.8  # 至少80%的依赖已安装
    )
    
    if overall_success:
        print(f"\n✅ 整体测试通过！系统可以正常使用。")
    else:
        print(f"\n⚠️  部分测试未通过，请检查以下问题:")
        
        if import_success < import_total:
            failed_imports = [name for name, success in import_results.items() if not success]
            print(f"   - 模块导入失败: {failed_imports}")
        
        if passed_tests < total_tests:
            failed_tests = [name for name, result in tests if not result]
            print(f"   - 功能测试失败: {failed_tests}")
        
        if installed_deps < total_deps * 0.8:
            missing_deps = [name for name, status in dependencies.items() if status == "未安装"]
            print(f"   - 缺少依赖包: {missing_deps}")
    
    print(f"\n建议:")
    if installed_deps < total_deps:
        print(f"   - 运行 'pip install -r requirements.txt' 安装缺少的依赖")
    if not any("cuda" in str(v).lower() for v in dependencies.values()):
        print(f"   - 考虑安装 GPU 版本的依赖以提高性能")
    print(f"   - 查看 example_modular.py 了解详细使用方法")


def main():
    """主测试函数。"""
    print("模块化RAG系统测试")
    print("时间:", "2025-10-29")
    
    try:
        # 执行各项测试
        import_results = test_imports()
        config_test = test_config_module()
        components_test = test_basic_components()
        dependencies = test_dependencies()
        pipeline_test = test_minimal_pipeline()
        
        # 生成报告
        generate_test_report(
            import_results,
            config_test, 
            components_test,
            dependencies,
            pipeline_test
        )
        
    except KeyboardInterrupt:
        print(f"\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生未预期错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()