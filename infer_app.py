import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import YOLO

def infer():
    model = YOLO('.\\8mpt\\best.pt')
    results = model('.\\imgs\\train_6676.jpg')
    print(results[0].plot())
    cv2.imwrite('./result.png', results[0].plot())

if __name__ == '__main__':
    infer()
