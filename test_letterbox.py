from ultralytics.data.augment import LetterBox
import numpy as np
import cv2

img = cv2.imread('./public/dataset/banana_anthracnose_Bing_0007.jpg')
lb = LetterBox((640, 640), auto=False, stride=32)
out = lb(image=img)
print("Original shape:", img.shape)
print("Padded shape:", out.shape)
print("Transforms / padded values:", out[0:5, 0:5])
