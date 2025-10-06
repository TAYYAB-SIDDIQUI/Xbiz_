import cv2

def preprocess(path):
    img=cv2.imread(path)
    blurred=cv2.GaussianBlur(img,(5,5),2)
    sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    denoised = cv2.fastNlMeansDenoisingColored(sharp, None, 6, 10, 9, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)
    
    return norm