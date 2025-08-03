#!/bin/bash
# 02_download_model.sh

echo "正在下载预训练模型..."

# 定义模型文件
pretrained_checkpoint='limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
base_pipeline_file='limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config'

# 设置 HOMEFOLDER (如果脚本不是从项目根目录执行，此行可确保路径正确)
export HOMEFOLDER=$(pwd) 

# 创建文件夹并下载
mkdir -p "${HOMEFOLDER}/models/mymodel/"
cd "${HOMEFOLDER}/models/mymodel/"

echo "下载预训练检查点: ${pretrained_checkpoint}"
wget "https://downloads.limelightvision.io/models/${pretrained_checkpoint}"
tar -xvf "${pretrained_checkpoint}"

echo "下载基础配置文件: ${base_pipeline_file}"
wget "https://downloads.limelightvision.io/models/${base_pipeline_file}"

# 返回项目根目录
cd "$HOMEFOLDER"

echo "模型下载完成。"
