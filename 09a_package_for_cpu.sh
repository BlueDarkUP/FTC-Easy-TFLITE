#!/bin/bash
# 09a_package_for_cpu.sh

echo "正在为 CPU/Control Hub 打包最终模型..."

# 确保在项目根目录
export HOMEFOLDER=$(pwd)
FINALOUTPUTFOLDER="${HOMEFOLDER}/final_output/"

# 复制标签和配置文件到最终输出目录
echo "复制标签和配置文件..."
cp "${HOMEFOLDER}/limelight_neural_detector_labels.txt" "$FINALOUTPUTFOLDER"
cp "${HOMEFOLDER}/models/mymodel/pipeline_file.config" "$FINALOUTPUTFOLDER"

# 进入 final_output 目录进行打包，方便路径管理
cd "$FINALOUTPUTFOLDER"

# 创建 zip 压缩包，-r 递归，-j 只存储文件名不存储路径
# 注意：我们这里不使用 -j 来保留 saved_model 的目录结构
echo "创建 control_hub_model.zip..."
zip -r "${HOMEFOLDER}/control_hub_model.zip" \
    limelight_neural_detector_8bit.tflite \
    limelight_neural_detector_32bit.tflite \
    limelight_neural_detector_labels.txt \
    pipeline_file.config \
    saved_model/

# 返回项目根目录
cd "$HOMEFOLDER"

echo "打包完成: ${HOMEFOLDER}/control_hub_model.zip"
