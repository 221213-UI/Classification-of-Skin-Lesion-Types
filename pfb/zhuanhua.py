import torch
import torchvision

# 1. 创建模型（不使用预训练权重）
model = torchvision.models.mobilenet_v3_large(pretrained=False)

# 2. 修改分类头，改成你的类别数（8类）
num_classes = 8  # 根据错误信息，你的模型是8类
model.classifier[3] = torch.nn.Linear(1280, num_classes)

# 3. 加载你训练好的权重
model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))

# 4. 设置为评估模式
model.eval()

# 5. 导出 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, 
    dummy_input, 
    "my_mobilenet.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print("导出成功！")
