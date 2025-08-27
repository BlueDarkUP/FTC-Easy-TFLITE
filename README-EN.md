### Before Reading: No TFRecord dataset yet?
  - #### Easily and smartly annotate data with [Zero2YoloYard](https://github.com/BlueDarkUP/Zero2YoloYard)
  - #### Easily convert YOLOv8 format datasets to TFRecord with [Yolo2TFRecord](https://github.com/BlueDarkUP/Yolo2TFRecord)

# Localized TFLite Object Detection Model Training Pipeline

This is a complete, end-to-end tutorial and executable pipeline designed to help you train your own object detection models locally within a Windows Subsystem for Linux (WSL) environment, leveraging an NVIDIA GPU, and ultimately generating a high-performance TFLite model that is **fully compliant with FIRST Tech Challenge (FTC) specifications and directly deployable**.

This pipeline is based on the TensorFlow 2 Object Detection API and uses a tested and stable version combination, integrated with TensorRT and TensorBoard monitoring, to ensure a professional, efficient, and reproducible development experience. Users simply need to clone this repository, prepare their dataset, and then execute the instructions in sequence.

---

## Table of Contents

*   [中文 README](README.md)
*   [Prerequisites](#prerequisites)
*   [Prerequisites](#prerequisites)
*   [Step One: Clone Repository and Professional Environment Setup](#step-one-clone-repository-and-professional-environment-setup)
*   [Step Two: Prepare Your Dataset](#step-two-prepare-your-dataset)
*   [Step Three: Model and Training Configuration](#step-three-model-and-training-configuration)
*   [Step Four: Start Training and Monitoring](#step-four-start-training-and-monitoring)
*   [Step Five: Export to TFLite Model](#step-five-export-to-tflite-model)
*   [Step Six: Model Quantization (INT8)](#step-six-model-quantization-int8)
*   [**Seventh Step: Verify Model Metadata (FTC Critical Step)**](#seventh-step-verify-model-metadata-ftc-critical-step)
*   [**Eighth Step: Package as FTC-Ready Model (Final Delivery)**](#eighth-step-package-as-ftc-ready-model-final-delivery)
*   [Appendix A: Starting a New Model Training (Workspace Cleanup)](#appendix-a-starting-a-new-model-training-workspace-cleanup)
*   [Appendix B: Generic Model Packaging Options](#appendix-b-generic-model-packaging-options)
*   [Troubleshooting (FAQ)](#troubleshooting-faq)

---

## Prerequisites

Before you begin, please ensure your system meets the following conditions:

1.  **Operating System**: Windows 10 or 11, with [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) installed.
2.  **Hardware**: An NVIDIA GPU (RTX series recommended, with at least 8GB VRAM).
3.  **Drivers**: Latest [NVIDIA Game Ready or Studio Drivers](https://www.nvidia.com/Download/index.aspx) installed on Windows.
4.  **Environment Management**: [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or Anaconda is recommended.

---

## Step One: Clone Repository and Professional Environment Setup

This step will download the project files, create a separate Python environment with TensorRT and GPU support, and install all necessary libraries and tools.

1.  **Open your WSL terminal**.

2.  **Clone this repository and enter the project directory**:
    We will perform all operations in the `/mnt/d/FTC_Training/` directory. You can modify the path as needed.
    ```bash
    # Switch to D drive
    cd /mnt/d/
    # Clone our repository
    git clone https://github.com/BlueDarkUP/FTC-Easy-TFLITE.git FTC_Training
    # Enter project directory
    cd FTC_Training
    # Set the project home directory as an environment variable for convenience
    export HOMEFOLDER=$(pwd)
    echo "Project home directory set to: $HOMEFOLDER"
    ```

3.  **Create and activate the Conda environment**:
    We will create an environment named `ftc_train`.
    ```bash
    conda create -n ftc_train python=3.10 -y
    conda activate ftc_train
    ```

4.  **Install TensorFlow, TensorRT, and CUDA (Core Step)**:
    We will install in two steps to ensure compatibility. First install TensorRT, then TensorFlow; `pip` will automatically handle version dependencies.
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

7.  **Verify the environment**:
    Run the following command. If successful, it will display your GPU information and eventually output `OK`.
    ```bash
    # Verify if TensorFlow can see the GPU
    python3 -c "import tensorflow as tf; print('GPU available: ', tf.config.list_physical_devices('GPU'))"
    
    # Run the official test script to verify API installation
    python3 ${HOMEFOLDER}/models/research/object_detection/builders/model_builder_tf2_test.py
    ```

---

## Step Two: Prepare Your Dataset

1.  **Obtain Data**:
    *   This project requires datasets in `TFRecord` format. We highly recommend using [Zero2YoloYard](https://github.com/BlueDarkUP/Zero2YoloYard) to annotate, manage, and export your data. When exporting, select the `TensorFlow TFRecord` format.
    *   You will receive a `.zip` file.

2.  **Place and Unzip Data**:
    *   Move your downloaded `dataset.zip` file to your project home directory (`D:\FTC_Training`).
    *   In the WSL terminal, unzip it:
        ```bash
        cd $HOMEFOLDER
        unzip -o dataset.zip -d $HOMEFOLDER
        ```

---

## Step Three: Model and Training Configuration

1.  **Automatically Locate File Paths**:
    The `01_find_paths.py` script provided in this repository will automatically find your data files.
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
    # Load number of classes into environment variables
    source class_vars.sh
    echo "Loaded number of classes: $num_classes"
    ```
    This script will also automatically generate `pipeline_file.config` based on your data.

---

## Step Four: Start Training and Monitoring

1.  **Apply Critical Compatibility Patch**:
    Due to library versions, a patch must be applied to avoid program crashes.
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

3.  ** (Optional) Start TensorBoard Monitoring**:
    To visualize the training process in real-time (e.g., changes in loss function), **open a new WSL terminal** and execute the following commands:
    ```bash
    # In the new terminal...
    # 1. Enter the project directory
    cd /mnt/d/FTC_Training/
    # 2. Activate the environment
    conda activate ftc_train
    # 3. Start TensorBoard, pointing to the training log directory
    tensorboard --logdir ./training_progress/
    ```
    Then, open the provided `http://localhost:6006/` link in your **Windows browser**.

4.  **Start Training in the Original Terminal!**
    ```bash
    # In the first terminal...
    # Set the final path variables
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
    Now, you should see the training logs begin to scroll. You can switch to the TensorBoard browser page at any time and click the refresh button in the top right corner to view the latest `loss` curve.

    You can press `Ctrl+C` at any time to terminate training prematurely. Model checkpoints will be saved in the `training_progress` folder.

---

## Step Five: Export to TFLite Model

After training is complete (or terminated prematurely), we will convert the TensorFlow checkpoint into a standard 32-bit floating-point TFLite model.

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

Quantization converts the model from 32-bit floating-point to 8-bit integer, significantly reducing model size (by approximately 4x) and increasing inference speed on the CPU.

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

## **Seventh Step: Verify Model Metadata (FTC Critical Step)**

Before deploying the model to the robot, we **must** verify that it contains the metadata required by the FTC SDK. Missing metadata is the **primary reason** for FTC App crashes (`task_vision_jni` error).

This repository provides a standalone diagnostic tool called `ftc_model_inspector.py`, which **does not require the `tflite-support` library** and can directly check if the model contains metadata.

1.  **Run the Inspection Script**:
    We will inspect the 8-bit quantized model just generated.
    ```bash
    python3 ftc_model_inspector.py final_output/limelight_neural_detector_8bit.tflite
    ```

2.  **Interpret the Results**:
    Since our model was just converted from a TensorFlow checkpoint, it **does not contain any metadata**. Therefore, you should see an **expected error** similar to the following:
    ```
    --- [Metadata Standalone Check] ---
    ❌ [Critical Issue] This .tflite file does not contain any metadata.
    ```
    **Seeing this result is perfectly normal!** It precisely demonstrates the necessity of the next packaging operation. It tells us that this "bare" model cannot be directly used on FTC yet.

---

## **Eighth Step: Package as FTC-Ready Model (Final Delivery)**

This is the **final and most crucial step** to transform your model into one that can be directly used on the Control Hub. We will use the `package_ftc_model_standalone.py` script, which will "glue" all necessary metadata (normalization parameters and label files) onto your model.

1.  **Run the Packaging Script**:
    This script will automatically find the required files and generate a new model compliant with FTC specifications.
    ```bash
    python3 package_ftc_model_standalone.py final_output/
    ```

2.  **Obtain the Final Product**:
    After the script executes successfully, a file named `limelight_neural_detector_8bit_ftc_ready.tflite` will be generated in the `final_output` folder.
    **This `_ftc_ready.tflite` file is the only model file you need to upload to the robot controller.**

3.  **(Optional but Highly Recommended) Final Verification**:
    To be absolutely sure, let's check this newly generated "FTC-ready" model again with the inspector.
    ```bash
    python3 ftc_model_inspector.py final_output/limelight_neural_detector_8bit_ftc_ready.tflite
    ```
    This time, you should see **all checks pass (`✅`)**! This confirms that the model now contains all the necessary metadata.


4.  Running the packaging script will create a file named `FTC_Deployment_Package.zip` in the `final_output/` folder. This ZIP package contains:
    *   The final `..._ftc_ready.tflite` model file.
    *   The `...labels.txt` label file, for programmer reference.
    *   The `pipeline_file.config` training configuration file, for traceability.

**Congratulations!** Now, you just need to hand this `FTC_Deployment_Package.zip` file to the robot programming team, and they will have everything needed for deployment and programming.

---

## **Appendix A: Starting a New Model Training (Workspace Cleanup)**

When you have successfully completed training and packaging a model and wish to start a brand new project (e.g., using a completely different dataset), performing the following cleanup steps is good practice. This ensures that old configuration files, datasets, and model checkpoints do not interfere with your new training.

**WARNING**: This operation will **permanently delete** your dataset, training progress, and all output files. Before running, ensure you have backed up any artifacts you wish to keep (e.g., `FTC_Deployment_Package.zip`).

### **Cleanup Steps**

1.  **Ensure you are in the project home directory**:
    ```bash
    # If you are unsure, execute this command
    cd /mnt/d/FTC_Training/
    ```

2.  **Run the One-Click Cleanup Script**:
    This repository provides a script named `10_clean_workspace.sh`, which will automatically delete all files and folders related to a specific training project.
    ```bash
    ./10_clean_workspace.sh
    ```

### **What does the cleanup script do?**

The `10_clean_workspace.sh` script deletes the following:

*   `final_output/`: Contains all exported `TFLite` models and `saved_model`s.
*   `training_progress/`: Contains all model checkpoints and TensorBoard logs.
*   `extracted_samples/`: Sample images used for quantization.
*   `train/`, `valid/`, `test/`: Dataset folders unzipped from `dataset.zip`.
*   `dataset.zip`: Your uploaded dataset archive.
*   `*.zip`: All packaged model archive files, e.g., `FTC_Deployment_Package.zip`.
*   `*.txt`, `*.config`, `*.sh`: All configuration files and label files generated by scripts.

### **How to start new training after cleanup?**

After running the cleanup script, your working directory will be restored to a "clean" state (only the core project scripts and the `models` folder remain).

You can directly start from **[Step Two: Prepare Your Dataset](#step-two-prepare-your-dataset)**, upload and unzip your **new** `dataset.zip` file, and then continue with all subsequent steps to train your next model.

---

## **Appendix B: Generic Model Packaging Options**

If you need to package for non-FTC platforms (e.g., Limelight), or require a generic package containing all model variants (32-bit, 8-bit, EdgeTPU), you can use the original packaging scripts provided in this repository.

**WARNING**: The `.tflite` files generated by these scripts **do not contain the metadata required by FTC** and cannot be used directly on the Control Hub!

### **Option A: Package for Generic CPU/GPU**
```bash
./09a_package_for_cpu.sh
```
This will create a `control_hub_model.zip` file containing 32-bit and 8-bit models, labels, and configuration files.

### **Option B: Package for Limelight (Google Coral)**
```bash
./09b_package_for_coral.sh
```
This will additionally install the Coral compiler, compile the 8-bit model into an `_edgetpu` version, and package all models.

---

## Troubleshooting (FAQ)
*   **Encountered `Segmentation fault (core dumped)`**:
    This usually indicates underlying library version conflicts. The most common cause is a `protobuf` version issue. Try `pip install --force-reinstall protobuf==3.20.3`. If the problem persists, ensure you strictly followed the library versions in **Step One**.

*   **`FileNotFoundError`**:
    Most commonly, `pipeline_file.config not found`. Please ensure you have successfully run the `03_generate_labels_and_config.py` script. If `.tfrecord` files are not found, check if your `dataset.zip` file structure is correct.

*   **Loss value becomes `NaN` after training starts**:
    This usually means the learning rate is too high. You can edit the `03_generate_labels_and_config.py` script, lower the `LEARNING_RATE` by an order of magnitude (e.g., from `.004` to `.0004`), then rerun the script and start training.

*   **`SyntaxError: from __future__ imports must occur at the beginning of the file`**:
    This indicates that the compatibility patch in **Step Four** was applied incorrectly. Please strictly follow the guide and run the `04_apply_tf_slim_patch.sh` script.

*   **TensorBoard interface shows "No dashboards are active"**:
    Don't worry. This usually means the training script has not yet written its first checkpoint. Please **wait a few minutes** and then refresh your browser page.
