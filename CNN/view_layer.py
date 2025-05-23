import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 添加在导入任何库之前

# 1. 定义模型类（必须与训练时的代码完全一致）
class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(320, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# 2. 实例化模型并加载参数
model = MnistModel()
model.load_state_dict(torch.load("./model/best_model.pkl", map_location=torch.device('cpu')))  # 假设在CPU上加载
model.eval()  # 设置为评估模式

# 定义与训练时相同的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor并归一化到 [0,1]
    # transforms.Normalize((0.1307,), (0.3081,))  # 如果训练时启用了归一化，取消注释此行
])

# 加载单张图片
image_path = "./dataset/mnist_test/train_0_7.png"  # 替换为你的实际路径
image = Image.open(image_path).convert('L')  # 'L' 表示转为灰度图

# 应用预处理
input_tensor = transform(image).unsqueeze(0)  # 添加批次维度 → [1, 1, 28, 28]


def visualize_conv1_features(model, input_tensor):
    # 注册钩子捕获 conv1 的输出
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu())

    hook = model.conv1.register_forward_hook(hook_fn)

    # 前向传播（触发钩子）
    with torch.no_grad():
        _ = model(input_tensor)

    # 移除钩子
    hook.remove()

    # 提取特征图（形状: [1, 10, 24, 24]）
    activations = feature_maps[0][0]  # 取第一个样本的10个通道

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(activations[i], cmap='viridis')
        ax.set_title(f"Channel {i + 1}")
        ax.axis('off')
    plt.suptitle("Conv1 Feature Maps")
    plt.show()


def visualize_layer_features(model, input_tensor, layer_name):
    # 获取目标层
    target_layer = dict(model.named_modules())[layer_name]

    # 注册钩子捕获输出
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu())

    hook = target_layer.register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    hook.remove()

    activations = feature_maps[0][0]  # [channels, H, W] 或 [features]

    # 可视化逻辑（适配不同层类型）
    if len(activations.shape) == 3:  # 卷积层特征图
        fig, axes = plt.subplots(4, 5, figsize=(15, 6))
        for i in range(min(20, activations.shape[0])):  # 最多显示10个通道
            ax = axes[i // 5, i % 5]
            ax.imshow(activations[i], cmap='viridis')
            ax.set_title(f"Channel {i + 1}")
            ax.axis('off')
    else:  # 全连接层等
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(activations)), activations.numpy())
        plt.title("Feature Vector")

    plt.suptitle(f"{layer_name} Activations")
    plt.show()


# 使用示例：可视化第二卷积层
visualize_layer_features(model, input_tensor, 'conv1')

# 调用函数
#visualize_conv1_features(model, input_tensor)
plt.show()