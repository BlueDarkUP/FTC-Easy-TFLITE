import tensorflow as tf
import os
import shutil

# 从环境变量获取路径
HOMEFOLDER = os.getenv('HOMEFOLDER', '/mnt/d/FTC_Training/')
FINALOUTPUTFOLDER = os.path.join(HOMEFOLDER, 'final_output/')

print("正在将 saved_model 转换为 32-bit TFLite 模型...")

# 1. 从导出的 SavedModel 加载转换器
converter = tf.lite.TFLiteConverter.from_saved_model(
    os.path.join(FINALOUTPUTFOLDER, 'saved_model')
)

# 2. 执行转换
tflite_model = converter.convert()

# 3. 保存 32 位模型
model_path_32bit = os.path.join(FINALOUTPUTFOLDER, 'limelight_neural_detector_32bit.tflite')
with open(model_path_32bit, 'wb') as f:
  f.write(tflite_model)

print(f"成功！32位 TFLite 模型已保存至: {model_path_32bit}")

# 4. 复制标签文件和配置文件到最终目录
try:
    shutil.copyfile(os.path.join(HOMEFOLDER, "limelight_neural_detector_labels.txt"), 
                    os.path.join(FINALOUTPUTFOLDER, "limelight_neural_detector_labels.txt"))
    shutil.copyfile(os.path.join(HOMEFOLDER, 'models/mymodel/pipeline_file.config'), 
                    os.path.join(FINALOUTPUTFOLDER, 'pipeline_file.config'))
    print("标签和配置文件已复制到 final_output 文件夹。")
except Exception as e:
    print(f"复制文件时出错: {e}")
