#!/bin/bash
# 05_export_inference_graph.sh

echo "正在导出推理图..."

# 确保在项目根目录
export HOMEFOLDER=$(pwd)

# 定义路径
export pipeline_file="${HOMEFOLDER}/models/mymodel/pipeline_file.config"
export model_dir="${HOMEFOLDER}/training_progress/"
export FINALOUTPUTFOLDER="${HOMEFOLDER}/final_output/"

# 清理并创建输出目录
rm -rf "$FINALOUTPUTFOLDER"
mkdir -p "$FINALOUTPUTFOLDER"

echo "导出模型到: $FINALOUTPUTFOLDER"
python3 "${HOMEFOLDER}/models/research/object_detection/export_tflite_graph_tf2.py" \
    --trained_checkpoint_dir "$model_dir" \
    --output_directory "$FINALOUTPUTFOLDER" \
    --pipeline_config_path "$pipeline_file"

echo "推理图导出完成。"

