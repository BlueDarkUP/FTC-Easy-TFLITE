# -*- coding: utf-8 -*-

"""
FTC .tflite 模型检查器 (全新方法)
=====================================

用途:
  检查 .tflite 模型文件，无需安装 'tflite-support' 库。
  此脚本直接读取模型文件，寻找附加的元数据，从而避免棘手的环境依赖问题。

原理:
  一个包含元数据的 .tflite 文件，其尾部会附加一个 ZIP 存档。
  此脚本会尝试以 ZIP 文件的形式打开模型，并读取其中的 'metadata.json' 文件
  来执行所有关键检查。

如何使用:
  1. 确保你的 Python 环境已安装 'tensorflow' (用于基础结构检查)。
  2. 运行脚本: python ftc_model_inspector.py /path/to/your/model.tflite
"""

import argparse
import os
import sys
import zipfile
import json

# 仍然使用 tensorflow 做基础的模型结构检查，因为它通常是可靠的
try:
    import tensorflow as tf
except ImportError:
    print("❌ 错误: 'tensorflow' 库未安装。")
    print("   此脚本需要 TensorFlow 来进行基础的模型结构检查。")
    print("   请根据您的需求安装，例如: pip install 'tensorflow[and-cuda]==2.15.0'")
    sys.exit(1)


def inspect_model(model_path: str):
    """
    对给定的 .tflite 模型文件执行一系列的验证检查。
    """
    if not os.path.exists(model_path):
        print(f"❌ 文件未找到: 无法在路径 '{model_path}' 找到模型文件。")
        return

    print("=" * 60)
    print(f"🔍 开始独立检查模型: {os.path.basename(model_path)}")
    print("=" * 60)

    # 标记，用于最终的总结
    all_checks_passed = True

    # --- 检查 1: TensorFlow Core 模型结构分析 (与之前相同) ---
    print("--- [检查 1/2] 模型基本结构 (使用 TensorFlow Core) ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        output_details = interpreter.get_output_details()
        num_outputs = len(output_details)
        expected_outputs = 4

        print(f"✅ [通过] TensorFlow Core 成功加载模型。")

        if num_outputs != expected_outputs:
            print(f"❌ [问题] 模型有 {num_outputs} 个输出，但 FTC 需要 {expected_outputs} 个。")
            all_checks_passed = False
        else:
            print(f"✅ [通过] 输出张量数量 ({num_outputs}) 正确。")

    except Exception as e:
        print(f"❌ [严重问题] 使用 TensorFlow Core 加载模型失败。")
        print(f"   错误详情: {e}")
        print("   结论：该模型文件已损坏或不是有效的 TFLite 文件。")
        print("=" * 60)
        return

    # --- 检查 2: 直接从 ZIP 存档中提取元数据 ---
    print("\n--- [检查 2/2] 元数据独立检查 (不依赖 tflite-support) ---")
    try:
        # 核心步骤：尝试像打开 zip 文件一样打开 .tflite 文件
        with zipfile.ZipFile(model_path, 'r') as z:
            # 列出所有元数据文件，用于调试
            file_list = z.namelist()
            print(f"   - 成功将模型作为 ZIP 打开。包含的元数据文件: {file_list}")

            # 检查 'metadata.json' 是否存在
            if 'metadata.json' not in file_list:
                print("❌ [严重问题] 模型中找到了 ZIP 存档，但缺少核心的 'metadata.json' 文件！")
                all_checks_passed = False
            else:
                print("✅ [通过] 找到了 'metadata.json' 文件。")
                with z.open('metadata.json') as f:
                    metadata = json.load(f)

                # a) 检查 NormalizationOptions
                has_normalization = False
                # 通常在第一个输入张量的 process_units 中
                try:
                    input_metadata = metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]
                    if 'process_units' in input_metadata:
                        for unit in input_metadata['process_units']:
                            if unit['options_type'] == 'NormalizationOptions':
                                has_normalization = True
                                mean = unit['options']['mean']
                                std = unit['options']['std']
                                print("✅ [通过] 找到了必需的 'NormalizationOptions' 元数据。")
                                print(f"   - 归一化均值 (Mean): {mean}")
                                print(f"   - 归一化标准差 (Std Dev): {std}")
                                break
                except (KeyError, IndexError):
                    pass  # 结构不匹配，下面会报告错误

                if not has_normalization:
                    print("❌ [问题] 'metadata.json' 中缺少 'NormalizationOptions'。")
                    print("   这是导致 'task_vision_jni' 错误的最常见原因。")
                    all_checks_passed = False

                # b) 检查标签文件
                has_labels = False
                try:
                    output_metadata = metadata['subgraph_metadata'][0]['output_tensor_metadata']
                    for tensor_meta in output_metadata:
                        if 'associated_files' in tensor_meta:
                            for file_info in tensor_meta['associated_files']:
                                if file_info['type'] == 'TENSOR_AXIS_LABELS':
                                    has_labels = True
                                    label_filename = file_info['name']
                                    print(f"✅ [通过] 找到了关联的标签文件: '{label_filename}'。")
                                    # 额外检查标签文件是否存在于 zip 中
                                    if label_filename not in file_list:
                                        print(f"   - ⚠️ [警告] 元数据提到了 '{label_filename}'，但该文件不在 ZIP 存档中。")
                                    break
                            if has_labels:
                                break
                except (KeyError, IndexError):
                    pass

                if not has_labels:
                    print("❌ [问题] 在元数据中找不到任何与输出关联的标签文件。")
                    all_checks_passed = False

    except zipfile.BadZipFile:
        print("❌ [严重问题] 此 .tflite 文件不包含任何元数据。")
        print("   (技术细节：文件尾部没有找到有效的 ZIP 存档)。")
        print("   修复建议：您必须在导出模型时使用 TFLite Metadata Writer API 添加元数据。")
        all_checks_passed = False
    except Exception as e:
        print(f"❌ [严重问题] 读取或解析元数据时发生未知错误: {e}")
        all_checks_passed = False

    # --- 最终结论 ---
    print("\n" + "=" * 60)
    print("📝 验证总结")
    print("=" * 60)
    if all_checks_passed:
        print("✅✅✅ 恭喜！您的模型通过了所有关键检查。")
        print("它很有可能与 FTC SDK 兼容。")
    else:
        print("❌❌❌ 注意！您的模型未能通过一项或多项关键检查。")
        print("直接使用此模型极有可能导致 FTC App 崩溃。")
        print("请回顾上面日志中的 [问题] 和 [警告] 部分，并修复您的模型导出流程。")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FTC .tflite 模型检查器 (独立版，无需 tflite-support)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="需要验证的 .tflite 模型的完整路径。"
    )
    args = parser.parse_args()
    inspect_model(args.model_path)
