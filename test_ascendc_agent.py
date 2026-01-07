#!/usr/bin/env python3
"""
测试AscendC算子Agent的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_generators.prompt_registry import PROMPT_REGISTRY
from dataset import dataset

def test_prompt_registry():
    """测试prompt注册器"""
    print("=== 测试Prompt注册器 ===")
    
    # 检查ascendc语言的prompt策略
    if "ascendc" in PROMPT_REGISTRY:
        strategies = PROMPT_REGISTRY["ascendc"]
        print(f"找到 {len(strategies)} 个AscendC prompt策略:")
        for strategy_name in strategies.keys():
            print(f"  - {strategy_name}")
    else:
        print("未找到AscendC prompt策略")
        return False
    
    return True

def test_operator_categories():
    """测试算子类别"""
    print("\n=== 测试算子类别 ===")
    
    # 统计各类型的算子数量
    category_counts = {}
    for op_name, op_info in dataset.items():
        category = op_info['category']
        if category not in category_counts:
            category_counts[category] = []
        category_counts[category].append(op_name)
    
    print("算子类别统计:")
    for category, ops in category_counts.items():
        print(f"  {category}: {len(ops)} 个算子")
        if len(ops) <= 5:  # 只显示前5个算子名称
            print(f"    {', '.join(ops)}")
        else:
            print(f"    {', '.join(ops[:5])}...")
    
    return category_counts

def test_prompt_generation():
    """测试prompt生成"""
    print("\n=== 测试Prompt生成 ===")
    
    # 选择几个代表性的算子进行测试
    test_operators = [
        "relu",           # 激活函数
        "layer_norm",     # 归一化
        "conv_standard_2d_square_input_square_kernel",  # 卷积
        "batched_matrix_multiplication",  # 矩阵乘法
        "add_bias_broadcast"  # 广播
    ]
    
    for op_name in test_operators:
        if op_name not in dataset:
            print(f"跳过不存在的算子: {op_name}")
            continue
            
        category = dataset[op_name]['category']
        print(f"\n测试算子: {op_name} (类别: {category})")
        
        # 测试不同的prompt策略
        strategies_to_test = [
            "add_shot",  # 基础策略
            "operator_specific",  # 通用算子指导
            "agent"  # 智能Agent
        ]
        
        # 根据类别添加特定策略
        if category == "activation":
            strategies_to_test.append("activation_specific")
        elif category == "normalization":
            strategies_to_test.append("normalization_specific")
        elif category == "convolution":
            strategies_to_test.append("convolution_specific")
        elif category == "matmul":
            strategies_to_test.append("matmul_specific")
        
        for strategy in strategies_to_test:
            if strategy in PROMPT_REGISTRY.get("ascendc", {}):
                try:
                    prompt_strategy = PROMPT_REGISTRY["ascendc"][strategy]
                    prompt = prompt_strategy.generate(op_name)
                    print(f"  ✓ {strategy}: 生成成功 (长度: {len(prompt)} 字符)")
                except Exception as e:
                    print(f"  ✗ {strategy}: 生成失败 - {e}")
            else:
                print(f"  - {strategy}: 策略不存在")

def test_api_reference():
    """测试API参考"""
    print("\n=== 测试API参考 ===")
    
    # 检查ascendc_agent中的API参考
    try:
        from prompt_generators.ascendc_agent import ASCENDC_API_REFERENCE
        print("AscendC API参考:")
        for category, apis in ASCENDC_API_REFERENCE.items():
            print(f"  {category}: {len(apis)} 个API")
            for api_name in apis.keys():
                print(f"    - {api_name}")
    except ImportError as e:
        print(f"无法导入API参考: {e}")

def main():
    """主测试函数"""
    print("AscendC算子Agent测试")
    print("=" * 50)
    
    # 运行各项测试
    test_prompt_registry()
    category_counts = test_operator_categories()
    test_prompt_generation()
    test_api_reference()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    
    # 总结
    total_operators = sum(len(ops) for ops in category_counts.values())
    print(f"\n总结:")
    print(f"- 总算子数量: {total_operators}")
    print(f"- 算子类别数量: {len(category_counts)}")
    print(f"- 支持的prompt策略: {len(PROMPT_REGISTRY.get('ascendc', {}))}")

if __name__ == "__main__":
    main()
