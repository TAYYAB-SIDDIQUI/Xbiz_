import cv2
import joblib
import numpy as np
import os
model=joblib.load("shape_detect_model (1).pkl")
def predict_shapes(filepath):
    img=cv2.imread(filepath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resize=cv2.resize(gray,(32,32))
    img_array = np.expand_dims(resize, axis=0)
    pr=model.predict(img_array)
    return pr
for i in os.listdir("test_samples"):
    ans=predict_shapes("test_samples"+"/"+i)
    if np.max(ans)<0.85:
        ans=-1
    else:
        ans=np.argmax(ans)
    if ans==0:
        cl="square"
    elif ans==1:
        cl="paralellogram"
    elif ans==2:
        cl="kite"
    elif ans==3:
        cl="rectangle"
    elif ans==4:
        cl="circle"
    elif ans==5:
        cl="triangle"
    elif ans==6:
        cl="trapezoid"
    elif ans==7:
        cl="rhombus"
    else:
        cl="unidentified"
    print(cl,i)
    