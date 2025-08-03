#!/bin/bash
# 04_apply_tf_slim_patch.sh

echo "正在应用 TF-Slim 兼容性补丁..."

# 确保在项目根目录
export HOMEFOLDER=$(pwd)

# 强制重装 tf-slim 以获取一个干净的文件
echo "强制重装 tf-slim 以重置文件..."
pip install --force-reinstall tf-slim

# 找到 tf-slim 库中需要修改的文件的路径
TF_SLIM_PATH=$(pip show tf-slim | grep Location | awk '{print $2}')/tf_slim/data/tfexample_decoder.py

# 自动应用修复
if [ -f "$TF_SLIM_PATH" ]; then
    echo "找到文件: $TF_SLIM_PATH"
    echo "正在插入 import tensorflow as tf ..."
    # 在第35行插入 import tensorflow as tf
    # 注意：如果文件行数少于35行，或者文件结构大变，此行可能需要调整
    sed -i '35iimport tensorflow as tf' "$TF_SLIM_PATH"
    
    echo "正在替换 control_flow_ops.case 为 tf.case ..."
    # 替换 control_flow_ops.case
    sed -i 's/control_flow_ops.case/tf.case/g' "$TF_SLIM_PATH"
    
    echo "正在替换 control_flow_ops.cond 为 tf.compat.v1.cond ..."
    # 替换 control_flow_ops.cond
    sed -i 's/control_flow_ops.cond/tf.compat.v1.cond/g' "$TF_SLIM_PATH"
    
    echo "补丁应用成功！"
else
    echo "错误: 未找到 tf_slim 文件 ($TF_SLIM_PATH)。请检查 tf-slim 是否已正确安装。"
    exit 1
fi
