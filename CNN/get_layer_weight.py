import torch
import scipy.io as sio
import numpy as np

model = torch.load("./model/best_model.pkl")
for name, param in model.items():
    layer_name = name.replace('.', '_')  # 处理层名特殊字符
    sio.savemat(f"{layer_name}.mat", {layer_name: param.cpu().numpy().astype(np.float64)})