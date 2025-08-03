import os
from object_detection.utils import label_map_util

HOMEFOLDER = os.getenv('HOMEFOLDER')
label_map_pbtxt_fname = os.getenv('label_map_pbtxt_fname')

def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    return len(label_map_util.create_category_index(categories).keys())

def get_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return [category['name'] for category in category_index.values()]

num_classes = get_num_classes(label_map_pbtxt_fname)
classes = get_classes(label_map_pbtxt_fname)

print('总类别数:', num_classes)
print('类别列表:', classes)

with open(os.path.join(HOMEFOLDER, "limelight_neural_detector_labels.txt"), 'w') as f:
    for label in classes:
        f.write(label + '\n')

with open(os.path.join(HOMEFOLDER, 'class_vars.sh'), 'w') as f:
    f.write(f'export num_classes={num_classes}\n')

print(f"\n标签文件 limelight_neural_detector_labels.txt 已创建。")
print(f"类别数量已保存到 class_vars.sh。")
