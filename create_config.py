import re
import os

# 从环境变量加载所有配置
HOMEFOLDER = os.getenv('HOMEFOLDER')
model_name = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
base_pipeline_file = 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config'

pipeline_fname = os.path.join(HOMEFOLDER, 'models/mymodel/', base_pipeline_file)
fine_tune_checkpoint = os.path.join(HOMEFOLDER, 'models/mymodel/', model_name, 'checkpoint/ckpt-0')

train_record_fname = os.getenv('train_record_fname')
val_record_fname = os.getenv('val_record_fname')
label_map_pbtxt_fname = os.getenv('label_map_pbtxt_fname')

batch_size = int(os.getenv('batch_size'))
num_steps = int(os.getenv('num_steps'))
num_classes = int(os.getenv('num_classes'))
chosen_model = 'ssd-mobilenet-v2'

print('正在生成自定义配置文件 pipeline_file.config ...')

with open(pipeline_fname) as f:
    s = f.read()

# 将所有路径和参数写入配置
s = re.sub('fine_tune_checkpoint: ".*?"', f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', f'input_path: "{train_record_fname}"', s)
s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', f'input_path: "{val_record_fname}"', s)
s = re.sub('label_map_path: ".*?"', f'label_map_path: "{label_map_pbtxt_fname}"', s)
s = re.sub('batch_size: [0-9]+', f'batch_size: {batch_size}', s)
s = re.sub('num_steps: [0-9]+', f'num_steps: {num_steps}', s)
s = re.sub('num_classes: [0-9]+', f'num_classes: {num_classes}', s)
s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "detection"', s)

if chosen_model == 'ssd-mobilenet-v2':
    s = re.sub('learning_rate_base: .8', 'learning_rate_base: .004', s)
    s = re.sub('warmup_learning_rate: 0.13333', 'warmup_learning_rate: .0016666', s)

# 输出最终的配置文件
output_config_path = os.path.join(HOMEFOLDER, 'models/mymodel/pipeline_file.config')
with open(output_config_path, 'w') as f:
    f.write(s)

print(f"最终配置文件已保存至: {output_config_path}")
