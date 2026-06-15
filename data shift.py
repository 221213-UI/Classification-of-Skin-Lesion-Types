import os
import pandas as pd
from shutil import copyfile
# 设置路径和参数
DATA_DIR = "ISIC_2019_Training_Input"  # 图像文件夹路径
CSV_PATH = "ISIC_2019_Training_GroundTruth.csv"  # 标注文件路径
NEW_DATA_DIR = "ISIC_2019_Classified"  # 新文件夹路径
# 1. 加载并处理标注文件
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # 提取类别列名
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    # 转换为单标签
    df['label'] = df[classes].idxmax(axis=1)
    return df
# 2. 创建新的文件夹结构并移动图像文件
def classify_images(df, data_dir, new_data_dir):
    # 确保新文件夹存在
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    # 为每个类别创建子文件夹
    for label in df['label'].unique():
        class_dir = os.path.join(new_data_dir, label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # 移动图像文件到对应的子文件夹中
    for index, row in df.iterrows():
        img_name = row['image'] + '.jpg'
        src_path = os.path.join(data_dir, img_name)
        dst_path = os.path.join(new_data_dir, row['label'], img_name)
        copyfile(src_path, dst_path)
    # 统计每个子目录中的图片数量
    image_counts = {}
    for label in df['label'].unique():
        class_dir = os.path.join(new_data_dir, label)
        image_counts[label] = len([name for name in os.listdir(class_dir) if name.endswith('.jpg')])

    return image_counts
# 主流程
if __name__ == "__main__":
    # 加载数据
    df = load_labels(CSV_PATH)
    # 根据标签分类并移动图像文件
    image_counts = classify_images(df, DATA_DIR, NEW_DATA_DIR)
    print(f"Images have been classified and moved to {NEW_DATA_DIR}")
    # 输出每个子目录中一共有多少图片
    for label, count in image_counts.items():
        print(f"{label}: {count} images")
