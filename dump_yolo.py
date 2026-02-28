try:
    from ultralytics.data.augment import LetterBox
    import numpy as np
    from PIL import Image

    l = LetterBox((640, 640), auto=False, scaleFill=False, stride=32)
    img = np.array(Image.open('./public/dataset/banana_anthracnose_Bing_0007.jpg'))
    out = l(image=img)
    print("LetterBox auto=False (standard ONNX): added pad", out.shape)
    
except Exception as e:
    print("Err")
