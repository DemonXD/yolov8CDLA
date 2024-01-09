import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if sys.platform in ["win32", "Linux"]:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(0)
    config_file = "dataset-win.yaml"
elif sys.platform == "darwin":
    device = torch.device("mps")
    config_file = "dataset-mac.yaml"

from ultralytics import YOLO

def train_model():
    # 加载模型
    model = YOLO("yolov8n.yaml").load("./8npt/best.pt")  # 使用预训练模型
    # print('model load。。。')
    # model = YOLO("./8npt/best.pt")  # 加载模型
    # print('model load completed。。。')

    print(" DEVICE: ", device)
    # 使用模型
    model.train(
        data=config_file,
        epochs=100,
        imgsz=640,
        device=device
        )  # 训练模型
    #
    metrics = model.val()  # 在验证集上评估模型性能
    #
    # print('metric : {}'.format(metrics))

    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

if __name__ == '__main__':
    train_model()
