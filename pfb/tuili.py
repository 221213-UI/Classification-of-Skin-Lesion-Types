import onnxruntime as ort
import numpy as np
from PIL import Image
import albumentations as A

# 1. 加载 ONNX 模型
session = ort.InferenceSession("my_mobilenet.onnx")

# 2. 预处理配置（与训练一致）
from torchvision.models import MobileNet_V3_Large_Weights
weights = MobileNet_V3_Large_Weights.DEFAULT
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])

# 3. 类别名称
class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# 4. 推理函数
def predict(image_path):
    # 读取和预处理
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = np.expand_dims(image_tensor, axis=0).astype(np.float32)
    
    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_tensor})
    logits = outputs[0]
    
    # 处理结果
    pred_class = np.argmax(logits, axis=1)[0]
    confidence = np.exp(logits[0, pred_class]) / np.sum(np.exp(logits[0])) * 100
    
    return class_names[pred_class], confidence

# 5. 使用
result, conf = predict('test.jpg')
print(f"预测: {result}, 置信度: {conf:.2f}%")
