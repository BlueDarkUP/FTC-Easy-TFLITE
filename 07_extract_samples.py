import tensorflow as tf
import os
import io
from PIL import Image
import shutil

# 从环境变量获取路径
HOMEFOLDER = os.getenv('HOMEFOLDER', '/mnt/d/FTC_Training/')
train_record_fname = os.getenv('train_record_fname')

# 定义样本输出目录
extracted_sample_folder = os.path.join(HOMEFOLDER, 'extracted_samples')

def extract_images_from_tfrecord(tfrecord_path, output_folder, num_samples=100):
    print(f"正在从 {tfrecord_path} 中提取样本图像...")
    # 如果目录存在，先删除
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    saved_images = 0
    # 使用 tf.data.TFRecordDataset 读取文件
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_record in raw_dataset.take(num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # 假设图像数据存储在 'image/encoded' 特征中
        image_data = example.features.feature['image/encoded'].bytes_list.value[0]
        
        # 解码并保存为 PNG 图像
        image = Image.open(io.BytesIO(image_data))
        image.save(os.path.join(output_folder, f'sample_{saved_images}.png'))
        
        saved_images += 1
        if saved_images >= num_samples:
            break
    
    print(f"成功提取 {saved_images} 张图像到 {output_folder}")

# 执行提取
if train_record_fname and os.path.exists(train_record_fname):
    extract_images_from_tfrecord(train_record_fname, extracted_sample_folder)
else:
    print(f"错误：找不到训练记录文件 at {train_record_fname}")
