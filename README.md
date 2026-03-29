# 🚀 FTC-Easy-TFLite Studio

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue.svg)]()
[![Backend](https://img.shields.io/badge/backend-WSL2%20%7C%20TF2-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

> ⚠️ **重要提示：全新 GUI 版本已发布！**
>
> 如果您只想轻松地训练专用于 FIRST Tech Challenge (FTC) 的物体检测模型，而**不想**通过命令行折腾底层的繁琐流水线或编写脚本，请直接前往本仓库的 **[Release](../../releases)** 页面。
>
> 下载最新发布的 `FTC-Easy-TFLite-Studio.exe`（全绿色单文件，无需安装），即可享受一键式的图形化训练部署体验。

---

## ✨ 简介

**FTC-Easy-TFLite Studio** 是专为 FIRST Tech Challenge (FTC) 竞赛队伍打造的现代化物体检测模型训练工作站。

我们将原本需要十几个步骤、极易出错的命令行训练流程，重新设计并封装进了一个优雅、顺滑且具有现代感（支持 Mica 毛玻璃特效与过渡动画）的图形用户界面（GUI）中。

它在后台静默接管 WSL2 和 TensorFlow 引擎，为您自动完成环境配置、数据解析、训练监控、INT8 量化以及最关键的 **FTC 元数据注入与打包**。让您能够专注于数据本身，而不是底层的技术细节。

## 🛠️ 核心特性

* **全图形化操作 (GUI)**：从环境一键构建到最终 Control Hub 部署包生成，全程鼠标点击完成。
* **丝滑交互体验**：内置精心设计的启动动画、页面切换动画以及实时滚动的数据看板（Loss/Step）。
* **WSL2 自动调度**：自动激活 Conda 环境、应用 TF补丁，无需手动打开 Linux 终端。
* **智能参数策略**：内置「快速验证」、「均衡模式」、「极致精度」三大一键预设，自动计算 Batch Size 与学习率黄金比例。
* **FTC 就绪 (FTC Ready)**：自动注入 FTC SDK 所需的所有元数据（归一化参数与标签），从源头彻底杜绝 `task_vision_jni` 崩溃。
* **一键式交付包**：直接输出包含模型、配置文件和标签文件的标准 ZIP 包，程序员U盘直插 Control Hub 即可使用。

## 🚀 快速上手 (致使用者)

1.  前往 **[Release](../../releases)** 页面下载 `FTC-Easy-TFLite-Studio.exe`。
2.  确保您的 Windows 已开启 **WSL2**，且电脑拥有 **NVIDIA GPU**。
3.  运行 .exe 文件，按照图形界面上的「环境」->「数据」->「训练」三个标签页顺序操作即可。

---

## 🏗️ 开发者与高级用户指南 (扩展原本流程)

本仓库仍然完整保留了底层的核心 Python 脚本和 Shell 脚本。如果您是一名开发者，或者希望对底层的 TensorFlow Object Detection 流水线进行深度定制、修改脚本逻辑、添加新的数据增强方法或探索更高级的训练配置，您完全可以继续使用旧的手动流程。

底层的脚本架构源自前一代的稳定工作流，我们鼓励通过以下途径进行扩展：

1.  **阅读旧版详细指南**：为了保持代码库的整洁，我们将您提供的那个详细的、包含八个步骤的本地化训练流水线 Markdown 文档移至了 **`Manual_Pipeline.md`**
2.  **修改底层脚本**：您可以直接修改仓库根目录下的 `01_find_paths.py` 到 `package_ftc_model_standalone.py` 等脚本。
3.  **提交 Pull Request**：如果您改进了底层的脚本逻辑，欢迎提交 PR，我们将把这些改进同步到 GUI 版本的后端调度中。

您可以从阅读 **[旧版手动流水线指南（旧 README）](Manual_Pipeline.md)** 开始您的自定义之旅。

## 📄 许可

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
