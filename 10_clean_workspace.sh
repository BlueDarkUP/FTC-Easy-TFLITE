#!/bin/bash
# 10_clean_workspace.sh

echo "####################################################################"
echo "# 警告：此操作将删除所有训练数据、进度和输出！                   #"
echo "# 这包括：                                                         #"
echo "# - final_output/, training_progress/, extracted_samples/          #"
echo "# - train/, valid/, test/, dataset.zip                             #"
echo "# - 所有生成的 .zip, .txt, .config, .sh (临时文件) 文件          #"
echo "#                                                                  #"
echo "# 在继续之前，请确保已备份所有重要产物。                           #"
echo "####################################################################"

read -p "您确定要继续吗？ (输入 'yes' 以确认): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "操作已取消。"
    exit 0
fi

echo "正在清理工作区..."

# 确保在项目根目录
export HOMEFOLDER=$(pwd)

# 删除由训练/导出过程生成的文件夹
rm -rf "${HOMEFOLDER}/final_output/"
rm -rf "${HOMEFOLDER}/training_progress/"
rm -rf "${HOMEFOLDER}/extracted_samples/"
rm -rf "${HOMEFOLDER}/train/"
rm -rf "${HOMEFOLDER}/valid/"
rm -rf "${HOMEFOLDER}/test/"

# 删除原始数据集压缩包和所有打包好的模型压缩包
rm -f "${HOMEFOLDER}/dataset.zip"
rm -f "${HOMEFOLDER}/*.zip" # 删除所有在根目录下的zip文件 (例如 control_hub_model.zip, limelight_models_all.zip)

# 删除由脚本生成的临时文件或配置文件
rm -f "${HOMEFOLDER}/limelight_neural_detector_labels.txt"
rm -f "${HOMEFOLDER}/path_vars.sh"
rm -f "${HOMEFOLDER}/class_vars.sh"
# 清理 models/mymodel/ 中可能存在的用户生成配置，保留原始下载的 .config 文件
rm -f "${HOMEFOLDER}/models/mymodel/pipeline_file.config"


echo "清理完成！您的工作区已重置。"
echo "您现在可以从 '第二步：准备您的数据集' 开始一个新的训练项目。"
