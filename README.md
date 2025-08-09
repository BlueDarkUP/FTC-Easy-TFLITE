### 阅读之前：还没有TFRecord数据集？
  - #### 使用[Zero2YoloYard](https://github.com/BlueDarkUP/Zero2YoloYard)轻松且智能的标注数据
  - #### 使用[Yolo2TFRecord](https://github.com/BlueDarkUP/Yolo2TFRecord)轻松的将YOLOv8格式数据集转换为TFRecord

# 本地化 TFLite 物体检测模型训练流水线

这是一个完整的、端到端的教学指南和可执行流水线，旨在帮助您在本地的 Windows Subsystem for Linux (WSL) 环境下，利用 NVIDIA GPU，训练自己的物体检测模型，并最终生成一个**完全符合 FIRST Tech Challenge (FTC) 规范的、可直接部署的**高性能 TFLite 模型。

本流水线基于 TensorFlow 2 Object Detection API，并使用了经过测试的稳定版本组合，同时集成了 TensorRT 和 TensorBoard 监控，以确保提供一个专业、高效且可复现的开发体验。用户只需克隆本仓库，准备好数据集，然后按顺序执行指令即可。

---

## 目录

*   [先决条件](#先决条件)
*   [第一步：克隆仓库与专业环境搭建](#第一步克隆仓库与专业环境搭建)
*   [第二步：准备您的数据集](#第二步准备您的数据集)
*   [第三步：模型与训练配置](#第三步模型与训练配置)
*   [第四步：开始训练与监控](#第四步开始训练与监控)
*   [第五步：导出为 TFLite 模型](#第五步导出为-tflite-模型)
*   [第六步：模型量化 (INT8)](#第六步模型量化-int8)
*   [**第七步：验证模型元数据 (FTC 关键步骤)**](#第七步验证模型元数据-ftc-关键步骤)
*   [**第八步：打包为 FTC 就绪模型 (最终交付)**](#第八步打包为-ftc-就绪模型-最终交付)
*   [附录 A：开启新模型的训练 (清理工作区)](#附录-a开启新模型的训练-清理工作区)
*   [附录 B：通用模型打包选项](#附录-b通用模型打包选项)
*   [故障排除 (FAQ)](#故障排除-faq)

---

## 先决条件

在开始之前，请确保您的系统满足以下条件：

1.  **操作系统**: Windows 10 或 11，并已安装 [WSL 2](https://learn.microsoft.com/zh-cn/windows/wsl/install)。
2.  **硬件**: 一块 NVIDIA GPU（建议使用 RTX 系列，至少 8GB 显存）。
3.  **驱动**: 在 Windows 上安装了最新的 [NVIDIA Game Ready 或 Studio 驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn)。
4.  **环境管理**: 推荐安装 [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 或 Anaconda。

---

## 第一步：克隆仓库与专业环境搭建

此步骤将下载本项目文件，创建包含 TensorRT 和 GPU 支持的独立 Python 环境，并安装所有必需的库和工具。

1.  **打开您的 WSL 终端**。

2.  **克隆本仓库并进入项目目录**：
    我们将所有操作都在 `/mnt/d/FTC_Training/` 目录下进行，您可以根据需要修改路径。
    ```bash
    # 切换到 D 盘
    cd /mnt/d/
    # 克隆我们的仓库
    git clone https://github.com/BlueDarkUP/FTC-Easy-TFLITE.git FTC_Training
    # 进入项目目录
    cd FTC_Training
    # 设置项目主目录为环境变量，方便后续使用
    export HOMEFOLDER=$(pwd)
    echo "项目主目录设置为: $HOMEFOLDER"
    ```

3.  **创建并激活 Conda 环境**：
    我们将创建一个名为 `ftc_train` 的环境。
    ```bash
    conda create -n ftc_train python=3.10 -y
    conda activate ftc_train
    ```

4.  **安装 TensorFlow、TensorRT 及 CUDA (核心步骤)**：
    我们将分两步安装，以确保兼容性。首先安装 TensorRT，然后安装 TensorFlow，`pip` 会自动处理版本依赖。
    ```bash
    # 1. 安装 TensorRT 库
    python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
    
    # 2. 安装 TensorFlow 2.15.0 及其兼容的 CUDA/cuDNN 库
    python3 -m pip install tensorflow[and-cuda]==2.15.0
    ```

5.  **克隆 TensorFlow Models 仓库并检出稳定版本**：
    ```bash
    rm -rf models
    git clone --depth 1 https://github.com/tensorflow/models.git
    cd models
    git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81
    git checkout ad1f7b56943998864db8f5db0706950e93bb7d81
    cd ..
    ```

6.  **安装 TensorFlow Object Detection API 依赖**：
    ```bash
    pip install protobuf==3.20.3
    
    # 编译 Protobuf 文件
    cd ${HOMEFOLDER}/models/research/
    protoc object_detection/protos/*.proto --python_out=.
    cd $HOMEFOLDER
    
    # 修改并安装 Object Detection API
    cp ${HOMEFOLDER}/models/research/object_detection/packages/tf2/setup.py ${HOMEFOLDER}/models/research/setup.py
    sed -i 's/tf-models-official>=2.5.1/tf-models-official==2.15.0/g' ${HOMEFOLDER}/models/research/setup.py
    pip install ${HOMEFOLDER}/models/research/
    ```

7.  **验证环境**：
    运行以下命令。如果成功，它会显示您的 GPU 信息，并最终输出 `OK`。
    ```bash
    # 验证 TensorFlow 能否看到 GPU
    python3 -c "import tensorflow as tf; print('GPU 可用: ', tf.config.list_physical_devices('GPU'))"
    
    # 运行官方测试脚本，验证 API 是否安装正确
    python3 ${HOMEFOLDER}/models/research/object_detection/builders/model_builder_tf2_test.py
    ```

---

## 第二步：准备您的数据集

1.  **获取数据**:
    *   本项目需要使用 `TFRecord` 格式的数据集。我们强烈推荐使用 [Roboflow](https://roboflow.com/) 来标注、管理和导出您的数据。在导出时，选择 `TensorFlow TFRecord` 格式。
    *   您将得到一个 `.zip` 文件。

2.  **放置并解压数据**:
    *   将您下载的 `dataset.zip` 文件移动到您的项目主目录 (`D:\FTC_Training`)。
    *   在 WSL 终端中，解压它：
        ```bash
        cd $HOMEFOLDER
        unzip -o dataset.zip -d $HOMEFOLDER
        ```

---

## 第三步：模型与训练配置

1.  **自动定位文件路径**:
    本仓库提供的 `01_find_paths.py` 脚本会自动查找您的数据文件。
    ```bash
    python3 01_find_paths.py
    # 将找到的路径加载到当前终端的环境变量中
    source path_vars.sh
    echo "标签文件: $label_map_pbtxt_fname"
    ```

2.  **下载预训练模型**:
    ```bash
    ./02_download_model.sh
    ```

3.  **设置训练超参数并生成配置文件**:
    您可以直接修改 `03_generate_labels_and_config.py` 脚本内的超参数。
    ```bash
    python3 03_generate_labels_and_config.py
    # 加载类别数量到环境变量
    source class_vars.sh
    echo "加载的类别数量: $num_classes"
    ```
    此脚本还会根据您的数据自动生成 `pipeline_file.config`。

---

## 第四步：开始训练与监控

1.  **应用关键的兼容性补丁**:
    由于库版本的原因，必须应用一个补丁来避免程序崩溃。
    ```bash
    ./04_apply_tf_slim_patch.sh
    ```
2.  **设置训练参数并生成配置文件 (关键步骤)**:
    可以在这里修改 batch_size 和 num_steps
    ```bash
    export batch_size=4
    export num_steps=10000
    echo "batch_size: $batch_size"
    echo "num_steps: $num_steps"
    python3 create_config.py
    echo "配置文件已生成: ${HOMEFOLDER}/models/mymodel/pipeline_file.config"
    ```

3.  **（可选）启动 TensorBoard 监控**：
    为了实时可视化训练过程（例如损失函数的变化），请**打开一个新的 WSL 终端**，并执行以下命令：
    ```bash
    # 在新终端中...
    # 1. 进入项目目录
    cd /mnt/d/FTC_Training/
    # 2. 激活环境
    conda activate ftc_train
    # 3. 启动 TensorBoard，指向训练日志目录
    tensorboard --logdir ./training_progress/
    ```    然后，在您的 **Windows 浏览器** 中打开它提供的 `http://localhost:6006/` 链接。

4.  **在原始终端中启动训练！**
    ```bash
    # 在第一个终端中...
    # 设置最后的路径变量
    export pipeline_file="${HOMEFOLDER}/models/mymodel/pipeline_file.config"
    export model_dir="${HOMEFOLDER}/training_progress/"

    # 清理上一次的训练进度（可选，但推荐）
    rm -rf $model_dir
    
    # 开始训练！
    python3 ${HOMEFOLDER}/models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path=${pipeline_file} \
        --model_dir=${model_dir} \
        --alsologtostderr
    ```
    现在，您应该能看到训练日志开始滚动。您可以随时切换到 TensorBoard 浏览器页面并点击右上角的刷新按钮来查看最新的 `loss` 曲线。

    您可以随时按 `Ctrl+C` 提前终止训练。模型检查点会保存在 `training_progress` 文件夹中。

---

## 第五步：导出为 TFLite 模型

训练完成后（或提前终止后），我们将把 TensorFlow 检查点转换为标准的 32位浮点 TFLite 模型。

1.  **导出推理图**:
    此脚本会自动选择 `training_progress/` 中最新的检查点。
    ```bash
    ./05_export_inference_graph.sh
    ```

2.  **转换为 TFLite**:
    ```bash
    python3 06_convert_to_tflite.py
    ```
    这将在 `final_output` 文件夹中生成 `limelight_neural_detector_32bit.tflite`。

---

## 第六步：模型量化 (INT8)

量化能将模型从 32位浮点数转换为 8位整数，显著减小模型体积（约4倍）并提升在 CPU 上的推理速度。

1.  **提取代表性数据集**:
    ```bash
    python3 07_extract_samples.py
    ```

2.  **执行 INT8 量化**:
    ```bash
    python3 08_quantize_model.py
    ```
    这将在 `final_output` 文件夹中生成 `limelight_neural_detector_8bit.tflite`。

---

## **第七步：验证模型元数据 (FTC 关键步骤)**

在将模型部署到机器人之前，我们**必须**验证它是否包含了 FTC SDK 所需的元数据。缺少元数据是导致 FTC App 崩溃（`task_vision_jni` 错误）的**首要原因**。

本仓库提供了一个名为 `ftc_model_inspector.py` 的独立诊断工具，它**无需 `tflite-support` 库**，可以直接检查模型是否包含元数据。

1.  **运行检查脚本**：
    我们将检查刚刚生成的 8位量化模型。
    ```bash
    python3 ftc_model_inspector.py final_output/limelight_neural_detector_8bit.tflite
    ```

2.  **解读结果**：
    由于我们的模型是刚刚从 TensorFlow 检查点转换而来的，它**不包含任何元数据**。因此，您应该会看到类似下面的**预期错误**：
    ```
    --- [元数据独立检查] ---
    ❌ [严重问题] 此 .tflite 文件不包含任何元数据。
    ```
    **看到这个结果是完全正常的！** 这恰恰证明了下一步打包操作的必要性。它告诉我们，这个“裸”模型还不能直接在 FTC 上使用。

---

## **第八步：打包为 FTC 就绪模型 (最终交付)**

这是将您的模型转化为可在 Control Hub 上直接使用的**最后一步，也是最重要的一步**。我们将使用 `package_ftc_model_standalone.py` 脚本，它会为您的模型“粘合”上所有必需的元数据（归一化参数和标签文件）。

1.  **运行打包脚本**：
    此脚本会自动查找所需的文件，并生成一个全新的、符合 FTC 规范的模型。
    ```bash
    python3 package_ftc_model_standalone.py final_output/
    ```

2.  **获取最终产物**：
    脚本执行成功后，会在 `final_output` 文件夹中生成一个名为 `limelight_neural_detector_8bit_ftc_ready.tflite` 的文件。
    **这个 `_ftc_ready.tflite` 文件，就是您唯一需要上传到机器人控制器上的模型文件。**

3.  **（可选但强烈推荐）最终验证**：
    为了百分百放心，让我们用检查器再次检查这个新生成的“FTC 就绪”模型。
    ```bash
    python3 ftc_model_inspector.py final_output/limelight_neural_detector_8bit_ftc_ready.tflite
    ```
    这一次，您应该会看到**所有检查全部通过 (`✅`)**！这证明模型已经包含了所有必需的元数据。


4.  运行打包脚本会在 `final_output/` 文件夹中创建一个名为 `FTC_Deployment_Package.zip` 的文件。这个 ZIP 包包含了：
    *   最终的 `..._ftc_ready.tflite` 模型文件。
    *   `...labels.txt` 标签文件，方便程序员参考。
    *   `pipeline_file.config` 训练配置文件，用于追溯。

**恭喜！** 现在，您只需将这个 `FTC_Deployment_Package.zip` 文件交给负责机器人编程的同学，他们就拥有了部署和编程所需的一切。

---

## **附录 A：开启新模型的训练 (清理工作区)**

当您成功完成一个模型的训练和打包，并希望开始一个全新的项目（例如，使用一个完全不同的数据集）时，执行以下清理步骤是一个好习惯。这可以确保旧的配置文件、数据集和模型检查点不会干扰到您的新训练。

**警告**：此操作会**永久删除**您的数据集、训练进度和所有输出文件。在运行之前，请确保您已经备份了需要保留的任何产物（例如 `FTC_Deployment_Package.zip`）。

### **清理步骤**

1.  **确保您在项目主目录中**：
    ```bash
    # 如果您不确定，请执行此命令
    cd /mnt/d/FTC_Training/
    ```

2.  **运行一键清理脚本**：
    本仓库提供了一个名为 `10_clean_workspace.sh` 的脚本，它会自动删除所有与特定训练项目相关的文件和文件夹。
    ```bash
    ./10_clean_workspace.sh
    ```

### **清理脚本会做什么？**

`10_clean_workspace.sh` 脚本会删除以下内容：

*   `final_output/`: 包含所有导出的 `TFLite` 模型和 `saved_model`。
*   `training_progress/`: 包含所有模型的检查点（checkpoints）和 TensorBoard 日志。
*   `extracted_samples/`: 用于量化的样本图像。
*   `train/`, `valid/`, `test/`: 从 `dataset.zip` 解压出的数据集文件夹。
*   `dataset.zip`: 您上传的数据集压缩包。
*   `*.zip`: 所有打包好的模型压缩文件，例如 `FTC_Deployment_Package.zip`。
*   `*.txt`, `*.config`, `*.sh`: 所有由脚本生成的配置文件和标签文件。

### **清理后如何开始新训练？**

在运行完清理脚本后，您的工作目录就恢复到了一个“干净”的状态（只剩下核心的项目脚本和 `models` 文件夹）。

您可以直接从 **[第二步：准备您的数据集](#第二步准备您的数据集)** 开始，上传并解压您的**新** `dataset.zip` 文件，然后继续执行后续的所有步骤，来训练您的下一个模型。

---

## **附录 B：通用模型打包选项**

如果您需要为非 FTC 的平台（例如 Limelight）打包，或者需要一个包含所有模型变体（32位、8位、EdgeTPU）的通用包，您可以使用本仓库提供的原始打包脚本。

**警告**：这些脚本生成的 `.tflite` 文件**不包含 FTC 所需的元数据**，不能直接在 Control Hub 上使用！

### **选项 A：为通用 CPU/GPU 打包**
```bash
./09a_package_for_cpu.sh
```
这将创建一个 `control_hub_model.zip` 文件，其中包含 32位和8位模型、标签和配置文件。

### **选项 B：为 Limelight (Google Coral) 打包**
```bash
./09b_package_for_coral.sh
```
这会额外安装 Coral 编译器，将 8位模型编译为 `_edgetpu` 版本，并打包所有模型。

---

## 故障排除 (FAQ)
*   **遇到 `Segmentation fault (core dumped)`**:
    这通常是底层库版本冲突。最常见的原因是 `protobuf` 版本问题。请尝试 `pip install --force-reinstall protobuf==3.20.3`。如果问题依旧，请确保严格按照**第一步**中的库版本进行安装。

*   **`FileNotFoundError`**:
    最常见的是 `pipeline_file.config not found`。请确保您已经成功运行了 `03_generate_labels_and_config.py` 脚本。如果找不到 `.tfrecord` 文件，请检查您的 `dataset.zip` 文件结构是否正确。

*   **训练开始后 Loss 值为 `NaN`**:
    这通常意味着学习率过高。您可以编辑 `03_generate_labels_and_config.py` 脚本，将 `LEARNING_RATE` 的值调低一个数量级（例如从 `.004` 改为 `.0004`），然后重新运行该脚本并开始训练。

*   **`SyntaxError: from __future__ imports must occur at the beginning of the file`**:
    这表示**第四步**中的兼容性补丁应用错误。请严格按照指南，运行 `04_apply_tf_slim_patch.sh` 脚本。

*   **TensorBoard 界面显示 "No dashboards are active"**:
    请不要担心。这通常意味着训练脚本还没有完成第一个检查点（checkpoint）的写入。请**等待几分钟**，然后刷新浏览器页面。
