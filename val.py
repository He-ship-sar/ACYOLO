import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
 model = YOLO("runs/ACyolo/train7/weights/best.pt")
 model.val(data="../datasets/SSDD82/ssdd.yaml", split='val', imgsz=640,
           batch=16, # rect=False,
           save_json=True, # 这个保存coco精度指标的开关
           # project='runs/test/ACmix-CCFM-MPDIoU',
           project='runs/ACyolo',
         
 )
