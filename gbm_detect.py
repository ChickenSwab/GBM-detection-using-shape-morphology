import cv2
import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import binary_dilation
import joblib

IMG_SIZE = 128

def extract_morph_features(mask):
    lbl = label(mask)
    props = regionprops(lbl)
    if props:
        r = props[0]
        area = r.area
        border = binary_dilation(mask) ^ mask
        perimeter_approx = np.count_nonzero(border)
        bbox = r.bbox
        bbox_volume = np.prod([bbox[i + 1] - bbox[i] for i in range(0, len(bbox), 2)])
        return [area, perimeter_approx, bbox_volume]
    else:
        return [0, 0, 0]

def preprocess_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Use Otsu's automatic thresholding
    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def predict_GBM(image_path):
    # Load the saved model (change path accordingly)
    model = joblib.load(r"D:\GBM_code\rf_brain_tumor_model.joblib")
    
    # Preprocess input image and extract features
    mask = preprocess_img(image_path)
    features = extract_morph_features(mask)
    
    # Prepare features for prediction and predict
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    
    return "yes" if pred == 1 else "no"

# Example usage:
if __name__ == "__main__":
    test_image_path = r"C:\Users\chand\Downloads\bren.jpeg"  # Replace with your image path
    result = predict_GBM(test_image_path)
    print(f"GBM detected? {result}")
