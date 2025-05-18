import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
 model = YOLO("runs/yolo11/HRSIDDATA_train/train2/weights/best.pt")
 model.val(data="../dataset/HRSID/hrsid.yaml", split='test', imgsz=640,
           batch=16, # rect=False,
           save_json=True, # 这个保存coco精度指标的开关
           # project='runs/test/ACmix-CCFM-MPDIoU',
           project='runs/yolo11/HRSIDDATA_test',
 )
