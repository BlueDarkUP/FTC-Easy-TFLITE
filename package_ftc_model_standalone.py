# -*- coding: utf-8 -*-

"""
FTC 模型部署包生成器 (最终版)
=====================================

用途:
  本脚本将一个标准的 TensorFlow Lite 模型导出目录，打包成一个符合
  FTC 规范的、包含完整交付物的单一 ZIP 部署包。

工作流程:
  1. 自动在指定目录中查找模型文件(.tflite)、标签文件(.txt)和配置文件(.config)。
  2. 手动为模型添加元数据，生成一个 FTC 就绪的 .tflite 文件。
  3. 创建一个名为 'FTC_Deployment_Package.zip' 的压缩文件。
  4. 将以下文件添加进 ZIP 包中:
     - 最终的 FTC 就绪模型 (..._ftc_ready.tflite)
     - 原始的标签文件 (..._labels.txt)
     - 原始的训练配置文件 (pipeline.config)
  5. 提供一个可直接交付的、完整的部署包。

如何使用:
  1. 确保您的 Python 环境仅安装了 'tensorflow' 库。
  2. 运行脚本: python create_ftc_deployment_package.py /path/to/your/dir
"""

import os
import sys
import argparse
import json
import zipfile
import io

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    print("❌ 错误：缺少 'tensorflow' 库。")
    print("   请运行： pip install tensorflow")
    sys.exit(1)


def find_required_files(directory: str) -> (str, str, str):
    """在指定目录中自动查找模型、标签和配置文件。"""
    tflite_files = [f for f in os.listdir(directory) if f.endswith(".tflite") and "ftc_ready" not in f]
    label_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    config_files = [f for f in os.listdir(directory) if f.endswith(".config")]

    if not tflite_files:
        print(f"❌ 错误：在目录 '{directory}' 中没有找到任何原始的 .tflite 模型文件。")
        return None, None, None
    
    if not label_files:
        print(f"❌ 错误：在目录 '{directory}' 中没有找到任何 .txt 标签文件。")
        return None, None, None

    model_file = next((f for f in tflite_files if "8bit" in f), tflite_files[0])
    label_file = label_files[0]
    config_file = config_files[0] if config_files else None

    print(f"✅ 成功找到文件：")
    print(f"   - 模型文件: {model_file}")
    print(f"   - 标签文件: {label_file}")
    if config_file:
        print(f"   - 配置文件: {config_file}")
    else:
        print("   - 警告: 未找到 .config 配置文件，打包时将跳过。")
    
    return (os.path.join(directory, model_file), 
            os.path.join(directory, label_file),
            os.path.join(directory, config_file) if config_file else None)


def create_metadata_json(input_details, label_file_name):
    """手动创建 metadata.json 的内容。 (与之前版本相同)"""
    input_dtype = input_details["dtype"]
    if input_dtype == np.uint8: mean, std = [127.5], [127.5]
    elif input_dtype == np.float32: mean, std = [0.0], [1.0]
    else: return None

    # (JSON 结构与之前版本完全相同，为简洁省略)
    metadata = {
      "name": "TFLite Object Detection Model for FTC", "description": "A model with manually attached metadata to be compatible with the FTC SDK.", "author": "Standalone Packager Script", "version": "1.0",
      "subgraph_metadata": [ { "input_tensor_metadata": [ { "name": "image", "description": "Input image to be detected.", "content": { "content_properties_type": "ImageProperties", "content_properties": { "color_space": "RGB" } }, "process_units": [ { "options_type": "NormalizationOptions", "options": { "mean": mean, "std": std } } ], "stats": { "max": [1.0], "min": [-1.0] } } ], "output_tensor_metadata": [ { "name": "location", "description": "The locations of the detected boxes.", "content": { "content_properties_type": "BoundingBoxProperties", "content_properties": {} } }, { "name": "category", "description": "The categories of the detected boxes.", "associated_files": [ { "name": os.path.basename(label_file_name), "description": "Labels for categories that the model can recognize.", "type": "TENSOR_AXIS_LABELS" } ] }, { "name": "score", "description": "The scores of the detected boxes." }, { "name": "number of detections", "description": "The number of detections." } ] } ]
    }
    return json.dumps(metadata, indent=2)


def create_deployment_package(input_dir: str):
    """主函数，执行完整的打包流程。"""
    print("=" * 60)
    print("📦 FTC 模型部署包生成器启动")
    print(f"📂 工作目录: {input_dir}")
    print("=" * 60)

    # 步骤 1: 查找文件
    print("\n--- [步骤 1/5] 正在查找输入文件... ---")
    model_path, label_path, config_path = find_required_files(input_dir)
    if not model_path: sys.exit(1)

    # 步骤 2: 创建元数据
    print("\n--- [步骤 2/5] 正在创建元数据... ---")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    metadata_json_content = create_metadata_json(interpreter.get_input_details()[0], label_path)
    if not metadata_json_content: sys.exit(1)
    print("✅ 成功创建 'metadata.json' 内容。")

    # 步骤 3: 构建元数据 ZIP 存档
    print("\n--- [步骤 3/5] 正在构建元数据 ZIP 存档... ---")
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zf:
        zf.writestr('metadata.json', metadata_json_content)
        zf.write(label_path, arcname=os.path.basename(label_path))
    print("✅ 成功创建内存中的元数据 ZIP。")

    # 步骤 4: 合并模型和元数据，生成 FTC 就绪文件
    print("\n--- [步骤 4/5] 正在生成 FTC 就绪模型... ---")
    with open(model_path, 'rb') as f: model_content = f.read()
    output_model_name = os.path.basename(model_path).replace(".tflite", "_ftc_ready.tflite")
    output_model_path = os.path.join(input_dir, output_model_name)
    with open(output_model_path, 'wb') as f:
        f.write(model_content)
        f.write(memory_zip.getvalue())
    print(f"✅ 成功生成 FTC 就绪模型: {output_model_name}")

    # 步骤 5: 创建最终的部署 ZIP 包
    print("\n--- [步骤 5/5] 正在创建最终部署包... ---")
    output_zip_path = os.path.join(input_dir, "FTC_Deployment_Package.zip")
    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        print(f"   -> 添加: {os.path.basename(output_model_path)}")
        zf.write(output_model_path, arcname=os.path.basename(output_model_path))
        
        print(f"   -> 添加: {os.path.basename(label_path)}")
        zf.write(label_path, arcname=os.path.basename(label_path))

        if config_path:
            print(f"   -> 添加: {os.path.basename(config_path)}")
            zf.write(config_path, arcname=os.path.basename(config_path))
    
    print(f"✅ 成功创建部署包: {os.path.basename(output_zip_path)}")

    print("\n" + "=" * 60)
    print("🎉 任务完成！🎉")
    print("=" * 60)
    print(f"一个完整的部署包 '{os.path.basename(output_zip_path)}' 已在您的目录中创建。")
    print("\n这个 ZIP 包包含了机器人编程所需的一切：")
    print(f"  ➡️  {os.path.basename(output_model_path)} (核心模型文件)")
    print(f"  ➡️  {os.path.basename(label_path)} (标签参考文件)")
    if config_path:
        print(f"  ➡️  {os.path.basename(config_path)} (训练配置文件)")
    
    print("\n🚀 请将这个 ZIP 文件直接交给负责机器人开发的同学！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTC 模型部署包生成器。")
    parser.add_argument("input_dir", type=str, help="包含 .tflite, .txt 和 .config 文件的目录。")
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        print(f"❌ 错误：目录不存在 -> '{args.input_dir}'")
        sys.exit(1)
    create_deployment_package(args.input_dir)
