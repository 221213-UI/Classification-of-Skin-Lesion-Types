import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
# 导入官方MobileNetV3Large和预训练权重
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import os
from PIL import Image
import albumentations as A
import numpy as np
import warnings

warnings.filterwarnings('ignore')   #屏蔽警告信息


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.data = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                # 过滤非图片文件，避免加载报错
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append((img_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, label


# 加载官方预训练权重
weights = MobileNet_V3_Large_Weights.DEFAULT
# 提取预训练模型的均值和标准差
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=mean, std=std)  # 替换为预训练配套归一化
])

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std)  # 测试集和训练集归一化保持一致
])
# 数据集根目录，替换为你实际的路径
root_dir = 'ISIC_2019_Classified'
# 加载自定义数据集
full_dataset = CustomDataset(root_dir=root_dir, transform=None)
# 固定随机种子，避免每次划分训练/测试集不一致
torch.manual_seed(42)
np.random.seed(42)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 应用不同的变换
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# 创建数据加载器：Windows下添加num_workers=0，避免多进程报错；pin_memory=True提升GPU训练速度
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(full_dataset.classes)  # 自动适配你的类别数（ISIC2019为8类）

# 加载官方预训练模型
model = mobilenet_v3_large(weights=weights)
# 替换最后一层全连接层，适配分类任务
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
# 将模型移到GPU/CPU
model = model.to(device)
# 解决ISIC2019类别不均衡：计算类别权重，加权交叉熵损失
class_counts = np.bincount([label for _, label in full_dataset.data])
class_weights = torch.tensor((1.0 / class_counts) * len(full_dataset.data) / num_classes, dtype=torch.float32).to(
    device)
criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加权损失，重视少样本类别

# 分层学习率优化器（微调核心：底层特征用小学习率保留预训练，顶层分类用大学习率适配任务）
optimizer = optim.AdamW(
    [
        {"params": model.features.parameters(), "lr": 1e-5},  # 底层特征层：极小学习率
        {"params": model.classifier.parameters(), "lr": 5e-4}  # 顶层分类层：正常学习率
    ],
    weight_decay=1e-4  # 权重衰减防止过拟合
)

num_epochs = 30  # 微调适当增加轮数，预训练模型收敛慢但效果好
best_accuracy = 0.0
print(f"训练设备：{device} | 类别数：{num_classes} | 训练集样本：{len(train_dataset)} | 测试集样本：{len(test_dataset)}")
print(f"预训练权重：{weights} | 模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    # 打印训练日志（保留你的格式，增加学习率显示）
    print(f'Epoch {epoch + 1:02d}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, '
          f'Test Accuracy: {test_accuracy:.2f}%, '
          f'LR(feat): {optimizer.param_groups[0]["lr"]:.6f}, LR(cls): {optimizer.param_groups[1]["lr"]:.6f}')

    # 保存最优模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        # 多卡/单卡兼容保存
        torch.save(model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                   'best_model.pth')
        print(f'✅ 最优模型已保存！当前最佳准确率：{best_accuracy:.2f}%')

# 训练完成打印最终结果
print(f'\n训练结束！最佳测试准确率：{best_accuracy:.2f}%，模型保存为：best_model.pth')