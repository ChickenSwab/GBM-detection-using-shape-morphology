import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
MODEL_PATH = r"D:\GBM_code\cnn_brain_tumor.h5"  # or .keras if you changed it

# Load trained model once
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    img = np.expand_dims(img, axis=0)  # (1, H, W, 1)
    return img

def predict_tumor(image_path, threshold=0.3, debug=True):
    x = preprocess_image(image_path)
    prob = float(model.predict(x, verbose=0)[0][0])  # probability of tumor

    label = "tumor" if prob >= threshold else "no_tumor"

    if debug:
        print(f"Image: {image_path}")
        print(f"Tumor probability = {prob:.3f}")
        print(f"Threshold = {threshold:.3f}")
        print(f"Predicted label = {label}")

    return label, prob

if __name__ == "__main__":
    test_image = r"C:\Users\chand\Downloads\bren5.png"  # change as needed
    label, p = predict_tumor(test_image, threshold=0.3)
    print(f"Final prediction: {label}, prob={p:.3f}")
