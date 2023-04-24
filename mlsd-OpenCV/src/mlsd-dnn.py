import cv2
import numpy as np



if __name__ == '__main__':
    net = cv2.dnn.readNetFromONNX('/home/cyh/mlsd-pointer-sim.onnx')
    print("load success.")

