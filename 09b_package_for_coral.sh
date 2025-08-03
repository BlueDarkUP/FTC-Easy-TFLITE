#!/bin/bash
# 09b_package_for_coral.sh

echo "正在为 Limelight (Google Coral EdgeTPU) 打包最终模型..."

# 确保在项目根目录
export HOMEFOLDER=$(pwd)
FINALOUTPUTFOLDER="${HOMEFOLDER}/final_output/"

# 1. 安装 Coral 编译器
echo "安装 Coral 编译器..."
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler -y

# 2. 编译模型
echo "正在为 EdgeTPU 编译 8-bit 模型..."
cd "$FINALOUTPUTFOLDER"
edgetpu_compiler limelight_neural_detector_8bit.tflite

# 重命名编译后的文件，符合原始笔记本的命名约定
if [ -f "limelight_neural_detector_8bit_edgetpu.tflite" ]; then
    mv limelight_neural_detector_8bit_edgetpu.tflite limelight_neural_detector_coral.tflite
    echo "模型已编译并重命名为 limelight_neural_detector_coral.tflite"
else
    echo "错误：编译失败，未找到 EdgeTPU 编译输出文件 (limelight_neural_detector_8bit_edgetpu.tflite)。"
    exit 1
fi

# 返回项目根目录
cd "$HOMEFOLDER"

# 3. 打包所有最终产物
echo "打包所有模型到 limelight_models_all.zip..."
# 复制标签和配置文件到最终输出目录
cp "${HOMEFOLDER}/limelight_neural_detector_labels.txt" "$FINALOUTPUTFOLDER"
cp "${HOMEFOLDER}/models/mymodel/pipeline_file.config" "$FINALOUTPUTFOLDER"

# 进入 final_output 目录进行打包
cd "$FINALOUTPUTFOLDER"

# 创建 zip 压缩包，包含所有最终模型和辅助文件
rm -f "${HOMEFOLDER}/limelight_models_all.zip" # 确保删除旧的
zip -r "${HOMEFOLDER}/limelight_models_all.zip" ./*

# 返回项目根目录
cd "$HOMEFOLDER"

echo "打包完成: ${HOMEFOLDER}/limelight_models_all.zip"
