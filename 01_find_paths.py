import os
import fnmatch

def find_files(directory, pattern):
    # 优先查找的目录
    preferred_dirs = ['train', 'valid', 'test']
    
    # 先在优先目录中查找
    for p_dir in preferred_dirs:
        search_path = os.path.join(directory, p_dir)
        if os.path.isdir(search_path):
            for root, _, files in os.walk(search_path):
                for basename in files:
                    if fnmatch.fnmatch(basename, pattern):
                        yield os.path.abspath(os.path.join(root, basename))
    
    # 如果没找到，再查找整个目录
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield os.path.abspath(os.path.join(root, basename))

def set_tfrecord_variables(directory):
    train_record_fname = ''
    val_record_fname = ''
    label_map_pbtxt_fname = ''

    for tfrecord_file in find_files(directory, '*.tfrecord'):
        if 'train' in tfrecord_file and not train_record_fname:
            train_record_fname = tfrecord_file
        elif 'valid' in tfrecord_file and not val_record_fname:
            val_record_fname = tfrecord_file

    for label_map_file in find_files(directory, '*_label_map.pbtxt'):
        label_map_pbtxt_fname = label_map_file
        break

    return train_record_fname, val_record_fname, label_map_pbtxt_fname

# --- 修改的部分 ---
# 从环境变量获取 HOMEFOLDER，如果不存在则使用当前目录 '.'
HOMEFOLDER = os.getenv('HOMEFOLDER', '.') 
# --- 修改结束 ---

train_record_fname, val_record_fname, label_map_pbtxt_fname = set_tfrecord_variables(HOMEFOLDER)

print("找到的训练记录文件:", train_record_fname)
print("找到的验证记录文件:", val_record_fname)
print("找到的标签映射文件:", label_map_pbtxt_fname)

# 将 path_vars.sh 文件生成在 HOMEFOLDER 目录下
with open(os.path.join(HOMEFOLDER, 'path_vars.sh'), 'w') as f:
    f.write(f'export train_record_fname="{train_record_fname}"\n')
    f.write(f'export val_record_fname="{val_record_fname}"\n')
    f.write(f'export label_map_pbtxt_fname="{label_map_pbtxt_fname}"\n')

print(f"\n文件路径已保存到 {os.path.join(HOMEFOLDER, 'path_vars.sh')}")
