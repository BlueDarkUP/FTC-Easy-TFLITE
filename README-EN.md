# Localized TFLite Object Detection Model Training Pipeline

This is a comprehensive, end-to-end tutorial and executable pipeline designed to help you train your own object detection models locally on a Windows Subsystem for Linux (WSL) environment, leveraging NVIDIA GPUs, and finally generate high-performance TFLite files ready for deployment.

This pipeline is based on the TensorFlow 2 Object Detection API, using a tested and stable version combination, and integrates TensorRT and TensorBoard monitoring to ensure a professional, efficient, and reproducible development experience. Users only need to clone this repository, prepare their dataset, and then follow the instructions in sequence.

---

## Table of Contents

*   [中文 README (Chinese README)](README.md)
*   [Prerequisites](#prerequisites)
*   [Step One: Clone Repository and Professional Environment Setup](#step-one-clone-repository-and-professional-environment-setup)
*   [Step Two: Prepare Your Dataset](#step-two-prepare-your-dataset)
*   [Step Three: Model and Training Configuration](#step-three-model-and-training-configuration)
*   [Step Four: Start Training and Monitoring](#step-four-start-training-and-monitoring)
*   [Step Five: Export as TFLite Model](#step-five-export-as-tflite-model)
*   [Step Six: Model Quantization (INT8)](#step-six-model-quantization-int8)
*   [Step Seven: Package Final Artifacts](#step-seven-package-final-artifacts)
*   [Appendix: Start Training a New Model (Clean Workspace)](#appendix-start-training-a-new-model-clean-workspace)
*   [Troubleshooting (FAQ)](#troubleshooting-faq)

---

## Prerequisites

Before you begin, ensure your system meets the following conditions:

1.  **Operating System**: Windows 10 or 11, with [WSL 2](https://learn.microsoft.com/zh-cn/windows/wsl/install) installed.
2.  **Hardware**: An NVIDIA GPU (RTX series recommended, at least 8GB VRAM).
3.  **Drivers**: Latest [NVIDIA Game Ready or Studio Drivers](https://www.nvidia.cn/Download/index.aspx?lang=cn) installed on Windows.
4.  **Environment Management**: [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or Anaconda recommended.

---

## Step One: Clone Repository and Professional Environment Setup

This step will download project files, create a standalone Python environment with TensorRT and GPU support, and install all necessary libraries and tools.

1.  **Open your WSL terminal**.

2.  **Clone this repository and navigate into the project directory**:
    We will perform all operations in the `/mnt/d/FTC_Training/` directory. You can modify the path as needed.
    ```bash
    # Switch to D drive
    cd /mnt/d/
    # Clone our repository
    git clone https://github.com/BlueDarkUP/FTC-Easy-TFLITE.git FTC_Training
    # Enter the project directory
    cd FTC_Training
    # Set the project home directory as an environment variable for later use
    export HOMEFOLDER=$(pwd)
    echo "Project home directory set to: $HOMEFOLDER"
    ```

3.  **Create and activate Conda environment**:
    We will create an environment named `ftc_train`.
    ```bash
    conda create -n ftc_train python=3.10 -y
    conda activate ftc_train
    ```

4.  **Install TensorFlow, TensorRT, and CUDA (Core Step)**:
    We will install in two steps to ensure compatibility. First install TensorRT, then TensorFlow, and `pip` will automatically handle version dependencies.
    ```bash
    # 1. Install TensorRT libraries
    python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
    
    # 2. Install TensorFlow 2.15.0 and its compatible CUDA/cuDNN libraries
    python3 -m pip install tensorflow[and-cuda]==2.15.0
    ```

5.  **Clone TensorFlow Models repository and checkout a stable version**:
    ```bash
    rm -rf models
    git clone --depth 1 https://github.com/tensorflow/models.git
    cd models
    git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81
    git checkout ad1f7b56943998864db8f5db0706950e93bb7d81
    cd ..
    ```

6.  **Install TensorFlow Object Detection API dependencies**:
    ```bash
    pip install protobuf==3.20.3
    
    # Compile Protobuf files
    cd ${HOMEFOLDER}/models/research/
    protoc object_detection/protos/*.proto --python_out=.
    cd $HOMEFOLDER
    
    # Modify and install Object Detection API
    cp ${HOMEFOLDER}/models/research/object_detection/packages/tf2/setup.py ${HOMEFOLDER}/models/research/setup.py
    sed -i 's/tf-models-official>=2.5.1/tf-models-official==2.15.0/g' ${HOMEFOLDER}/models/research/setup.py
    pip install ${HOMEFOLDER}/models/research/
    ```

7.  **Verify Environment**:
    Run the following commands. If successful, it will display your GPU information and finally output `OK`.
    ```bash
    # Verify if TensorFlow can see the GPU
    python3 -c "import tensorflow as tf; print('GPU Available: ', tf.config.list_physical_devices('GPU'))"
    
    # Run official test script to verify API installation
    python3 ${HOMEFOLDER}/models/research/object_detection/builders/model_builder_tf2_test.py
    ```

---

## Step Two: Prepare Your Dataset

1.  **Obtain Data**:
    *   This project requires `TFRecord` format datasets. We strongly recommend using [Roboflow](https://roboflow.com/) to annotate, manage, and export your data. When exporting, choose `TensorFlow TFRecord` format.
    *   You will receive a `.zip` file.

2.  **Place and Extract Data**:
    *   Move your downloaded `dataset.zip` file to your project home directory (`D:\FTC_Training`).
    *   In the WSL terminal, extract it:
        ```bash
        cd $HOMEFOLDER
        unzip -o dataset.zip -d $HOMEFOLDER
        ```

---

## Step Three: Model and Training Configuration

1.  **Automatically Locate File Paths**:
    The `01_find_paths.py` script provided in this repository will automatically locate your data files.
    ```bash
    python3 01_find_paths.py
    # Load the found paths into the current terminal's environment variables
    source path_vars.sh
    echo "Label file: $label_map_pbtxt_fname"
    ```

2.  **Download Pre-trained Model**:
    ```bash
    ./02_download_model.sh
    ```

3.  **Set Training Hyperparameters and Generate Configuration File**:
    You can directly modify the hyperparameters within the `03_generate_labels_and_config.py` script.
    ```bash
    python3 03_generate_labels_and_config.py
    # Load the number of classes into environment variables
    source class_vars.sh
    echo "Loaded number of classes: $num_classes"
    ```
    This script will also automatically generate `pipeline_file.config` based on your data.

---

## Step Four: Start Training and Monitoring

1.  **Apply Critical Compatibility Patch**:
    Due to library versioning, a patch must be applied to prevent crashes.
    ```bash
    ./04_apply_tf_slim_patch.sh
    ```
2.  **Set Training Parameters and Generate Configuration File (Critical Step)**:
    You can modify `batch_size` and `num_steps` here.
    ```bash
    export batch_size=4
    export num_steps=10000
    echo "batch_size: $batch_size"
    echo "num_steps: $num_steps"
    python3 create_config.py
    echo "Configuration file generated: ${HOMEFOLDER}/models/mymodel/pipeline_file.config"
    ```

3.  **(Optional) Start TensorBoard Monitoring**:
    To visualize the training process in real-time (e.g., loss curve changes), **open a new WSL terminal** and execute the following commands:
    ```bash
    # In the new terminal...
    # 1. Navigate to project directory
    cd /mnt/d/FTC_Training/
    # 2. Activate environment
    conda activate ftc_train
    # 3. Start TensorBoard, pointing to the training log directory
    tensorboard --logdir ./training_progress/
    ```
    Then, open the provided `http://localhost:6006/` link in your **Windows browser**.

4.  **Start Training in the original terminal!**
    ```bash
    # In the first terminal...
    # Set final path variables
    export pipeline_file="${HOMEFOLDER}/models/mymodel/pipeline_file.config"
    export model_dir="${HOMEFOLDER}/training_progress/"

    # Clean up previous training progress (optional, but recommended)
    rm -rf $model_dir
    
    # Start training!
    python3 ${HOMEFOLDER}/models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path=${pipeline_file} \
        --model_dir=${model_dir} \
        --alsologtostderr
    ```
    You should now see training logs scrolling. You can switch to the TensorBoard browser page at any time and click the refresh button in the top right corner to view the latest `loss` curve.

    You can press `Ctrl+C` at any time to terminate training early. Model checkpoints will be saved in the `training_progress` folder.

---

## Step Five: Export as TFLite Model

After training is complete (or terminated early), we will convert the TensorFlow checkpoint into a standard 32-bit floating-point TFLite model.

1.  **Export Inference Graph**:
    This script will automatically select the latest checkpoint from `training_progress/`.
    ```bash
    ./05_export_inference_graph.sh
    ```

2.  **Convert to TFLite**:
    ```bash
    python3 06_convert_to_tflite.py
    ```
    This will generate `limelight_neural_detector_32bit.tflite` in the `final_output` folder.

---

## Step Six: Model Quantization (INT8)

Quantization converts models from 32-bit floating-point to 8-bit integers, significantly reducing model size (approx. 4x) and improving inference speed on CPUs.

1.  **Extract Representative Dataset**:
    ```bash
    python3 07_extract_samples.py
    ```

2.  **Perform INT8 Quantization**:
    ```bash
    python3 08_quantize_model.py
    ```
    This will generate `limelight_neural_detector_8bit.tflite` in the `final_output` folder.

---

## Step Seven: Package Final Artifacts

Choose the appropriate packaging method based on your deployment target.

### **Option A: Package for Control Hub (CPU/GPU)**

This is your primary target. We will package the most important files.
```bash
./09a_package_for_cpu.sh
```
This will create a file named `control_hub_model.zip` containing:
*   `limelight_neural_detector_8bit.tflite` (preferred for CPU inference)
*   `limelight_neural_detector_32bit.tflite` (high-quality backup)
*   `limelight_neural_detector_labels.txt` (essential)
*   `pipeline_file.config` (for traceability)
*   `saved_model/` folder (for potential future TensorRT optimization)

### **Option B: Package for Limelight (Google Coral)**

If you also want to use it on Limelight with a Coral TPU, follow this step.
```bash
./09b_package_for_coral.sh
```
This will additionally install the Coral compiler, compile the `8bit` model to an `_edgetpu` version, and package all models.

**Congratulations!** Your model is now ready for deployment.

---

## Appendix: Start Training a New Model (Clean Workspace)

When you have successfully completed training and packaging a model and wish to start a brand new project (e.g., with a completely different dataset), it's good practice to perform the following cleanup steps. This ensures that old configuration files, datasets, and model checkpoints do not interfere with your new training.

**WARNING**: This operation will **permanently delete** your dataset, training progress, and all output files. Before running, ensure you have backed up any artifacts you wish to keep (e.g., `control_hub_model.zip`).

### **Cleanup Steps**

1.  **Ensure you are in the project home directory**:
    ```bash
    # If you are unsure, execute this command
    cd /mnt/d/FTC_Training/
    ```

2.  **Run the one-click cleanup script**:
    This repository provides a script named `10_clean_workspace.sh` which automatically deletes all files and folders related to a specific training project.
    ```bash
    ./10_clean_workspace.sh
    ```

### **What does the cleanup script do?**

The `10_clean_workspace.sh` script will delete the following:

*   `final_output/`: Contains all exported `TFLite` models and `saved_model`.
*   `training_progress/`: Contains all model checkpoints and TensorBoard logs.
*   `extracted_samples/`: Sample images used for quantization.
*   `train/`, `valid/`, `test/`: Dataset folders extracted from `dataset.zip`.
*   `dataset.zip`: Your uploaded dataset archive.
*   `*.zip`: All packaged model archives, such as `control_hub_model.zip`.
*   `*.txt`, `*.config`, `*.sh`: All configuration files and label files generated by scripts.

### **How to start new training after cleanup?**

After running the cleanup script, your working directory will be restored to a "clean" state (only core project scripts and the `models` folder remain).

You can directly start from **[Step Two: Prepare Your Dataset](#step-two-prepare-your-dataset)**, upload and extract your **new** `dataset.zip` file, and then proceed with all subsequent steps to train your next model.

---

## Troubleshooting (FAQ)
*   **Encountering `Segmentation fault (core dumped)`**:
    This is usually due to underlying library version conflicts. The most common cause is `protobuf` version issues. Try `pip install --force-reinstall protobuf==3.20.3`. If the issue persists, ensure you strictly followed the library versions in **Step One**.

*   **`FileNotFoundError`**:
    Most commonly, `pipeline_file.config not found`. Ensure you have successfully run the `03_generate_labels_and_config.py` script. If `.tfrecord` files are not found, check if your `dataset.zip` file structure is correct.

*   **Loss value is `NaN` after training starts**:
    This usually means the learning rate is too high. You can edit the `03_generate_labels_and_config.py` script to lower the `LEARNING_RATE` by an order of magnitude (e.g., from `.004` to `.0004`), then rerun the script and start training.

*   **`SyntaxError: from __future__ imports must occur at the beginning of the file`**:
    This indicates that the compatibility patch in **Step Four** was applied incorrectly. Please strictly follow the guide and run the `04_apply_tf_slim_patch.sh` script.

*   **TensorBoard interface shows "No dashboards are active"**:
    Don't worry. This usually means the training script has not yet written the first checkpoint. Please **wait a few minutes** and then refresh your browser page.
