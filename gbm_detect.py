import cv2
import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import binary_dilation
import joblib

IMG_SIZE = 128

def extract_morph_features(mask):
    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return [0.0, 0.0, 0.0]

    r = props[0]
    area = float(r.area)
    
    # Safe perimeter (works 2D/3D)
    border = binary_dilation(mask) ^ mask
    perimeter_approx = float(np.count_nonzero(border))
    
    # FIXED bbox volume calculation
    bbox = r.bbox
    if len(bbox) < 4 or len(bbox) % 2 != 0:
        bbox_volume = float(area * 4)  # Fallback
    else:
        bbox_sizes = []
        for i in range(0, len(bbox), 2):
            size = max(1, bbox[i + 1] - bbox[i])  # Ensure positive, min 1
            bbox_sizes.append(size)
        bbox_volume = float(np.prod(bbox_sizes))
    
    return [area, perimeter_approx, bbox_volume]



def preprocess_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.medianBlur(img, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ADD THIS FUNCTION HERE:
def is_mask_valid(mask, min_area=50):
    fg = np.count_nonzero(mask)
    if fg < min_area:
        return False
    if fg > 0.9 * mask.size:
        return False
    return True

def predict_GBM(image_path, debug=True):
    model = joblib.load(r"D:\GBM_code\rf_brain_tumor_model.joblib")
    mask = preprocess_img(image_path)

    if not is_mask_valid(mask):
        return "uncertain"

    features = extract_morph_features(mask)
    features = np.array(features).reshape(1, -1)
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    if debug:
        print(f"Features: area={features[0][0]:.0f}, perimeter={features[0][1]:.0f}, bbox_vol={features[0][2]:.0f}")
        print(f"Prediction: {'yes' if pred == 1 else 'no'} (prob: {prob[1]:.3f})")
    
    return "yes" if pred == 1 else "no"


# Example usage:
if __name__ == "__main__":
    test_image_path = r"C:\Users\chand\Downloads\nigger.jpeg"  # Replace with your image path
    result = predict_GBM(test_image_path)
    print(f"GBM detected? {result}")
