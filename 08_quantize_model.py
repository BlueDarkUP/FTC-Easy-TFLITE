import tensorflow as tf
import os
import glob
import random

# 从环境变量获取路径
HOMEFOLDER = os.getenv('HOMEFOLDER', '/mnt/d/FTC_Training/')
FINALOUTPUTFOLDER = os.path.join(HOMEFOLDER, 'final_output/')
model_path_32bit = os.path.join(FINALOUTPUTFOLDER, 'limelight_neural_detector_32bit.tflite')
extracted_sample_folder = os.path.join(HOMEFOLDER, 'extracted_samples')

# 1. 准备代表性数据集
quant_image_list = glob.glob(os.path.join(extracted_sample_folder, '*.png'))
print(f"找到 {len(quant_image_list)} 张样本图像用于量化。")

interpreter = tf.lite.Interpreter(model_path=model_path_32bit)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# --- 修正后的数据生成器 ---
def representative_data_gen():
    dataset_list = quant_image_list
    for i in range(100):
        pick_me = random.choice(dataset_list)
        image = tf.io.read_file(pick_me)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, [height, width])
        # 将图像归一化到 [0, 1] 范围并确保类型为 float32
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]
# --------------------------

# 2. 配置并运行转换器
print("正在初始化转换器以进行 INT8 量化...")
converter = tf.lite.TFLiteConverter.from_saved_model(
    os.path.join(FINALOUTPUTFOLDER, 'saved_model')
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# 确保只使用 INT8 支持的操作
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 设置最终量化模型的输入和输出类型
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

print("开始量化转换... 这可能需要几分钟时间。")
tflite_quant_model = converter.convert()
print("量化完成。")

# 3. 保存量化后的模型
quant_model_path = os.path.join(FINALOUTPUTFOLDER, 'limelight_neural_detector_8bit.tflite')
with open(quant_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"成功！8位量化 TFLite 模型已保存至: {quant_model_path}")
