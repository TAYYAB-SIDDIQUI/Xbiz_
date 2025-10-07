import cv2
import numpy as np
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from xgboost import XGBClassifier
import joblib

# # Paths
# data_dir = "/content/drive/MyDrive/Img"
# classes = os.listdir(data_dir)  # e.g., ['cat', 'dog', 'car']

# # Parameters
# img_size = (300, 300)  # small for fast processing
# radius = 2
# n_points = 8 * radius

# X, y = [], []

# for label, cls in enumerate(classes):
#     folder = os.path.join(data_dir, cls)
#     for file in os.listdir(folder):
#         img_path = os.path.join(folder, file)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         img = cv2.resize(img, img_size)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # HOG features
#         hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

#         # Color histogram
#         hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
#                             [0, 256, 0, 256, 0, 256])
#         hist = cv2.normalize(hist, hist).flatten()

#         # LBP features (texture)
#         lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
#         lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
#         lbp_hist = lbp_hist.astype("float")
#         lbp_hist /= (lbp_hist.sum() + 1e-6)

#         # Combine features
#         features = np.hstack([hog_features, hist, lbp_hist])
#         X.append(features)
#         y.append(label)

# X = np.array(X)
# y = np.array(y)

# print("Feature matrix shape:", X.shape)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Choose model
# #model = XGBClassifier()
# model = LogisticRegression(max_iter=1000)

# model.fit(X, y)
# y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred))
classes=['Circle', 'Square', 'Rectangle', 'Trapezoid', 'Semi circle', 'Parallelogram', 'Star', 'Pentagon','Triangle']
DATA_DIR = "data"   # Each subfolder = one class
MODEL_PATH = "image_model.pkl"
IMG_SIZE = (300,300)
RADIUS = 2
N_POINTS = 8 * RADIUS
TEST_SIZE = 0.2
USE_LOGISTIC = False
def extract_features(img_path):
    """Extract combined HOG + color histogram + LBP features from one image"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG (shape)
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Color histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP (texture)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.hstack([hog_features, hist, lbp_hist])
model=joblib.load(MODEL_PATH)
def predict_image(img_path):
    feats = extract_features(img_path)
    if feats is None:
        print("⚠️ Could not read image.")
        return None
    pred = model.predict([feats])[0]
    print(f"🖼️ Prediction for '{img_path}': {classes[pred]}")
    return classes[pred]
for i in os.listdir("test_samples"):
  print(predict_image("Test_samples"+"/"+i))