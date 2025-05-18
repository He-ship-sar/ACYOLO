import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-ACmix-CCFM.yaml')  # 从YAML建立一个新模型
    
    # 设置优化器参数
    optimizer_params = {
       # 'lr0': 0.007,  # 初始学习率
       # 'momentum': 0.937,
        # 如果您还需要设置其他优化器参数，可以在这里添加
        # 'weight_decay': 0.0005,
    }
    
    # 训练模型，并传递优化器参数
    results = model.train(
        data='../datasets/SSDD82/ssdd.yaml',
        # data='../dataset/HRSID/hrsid.yaml',
        # close_mosaic=0,
        epochs=150,
        imgsz=640,
        device=0,  # 使用GPU设备，如果有多个GPU，可以使用如 'cuda:1' 这样的指定
        optimizer='SGD',
        batch=16,
        # seed=8888,
        # project='runs/train-Test/ACmix-CCFM-InnerWIoU',
        # project='runs/IoU/ACmix-CCFM-MPDIoU',
        # project='runs/C3K2_MDConv-1-ACmix-CCFM',
        # project='runs/Yolov11',
        project='runs/ACyolo',
        amp=False,  # 是否使用混合精度训练
       # **optimizer_params  # 使用**kwargs传递优化器参数
    )
