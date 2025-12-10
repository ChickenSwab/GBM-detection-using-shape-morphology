# train_rf_on_combined_minimal.py
import os, time
import cv2
import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.measure import regionprops, label
from scipy.ndimage import binary_dilation

# ---------- CONFIG ----------
ROOT_DIR = r"D:\GBM_code_new\archive\Training"
CNN_MODEL_PATH = r"D:\GBM_code_new\cnn_brain_tumor.h5"   # or .keras
RF_SAVE_PATH = r"D:\GBM_code_new\rf_brain_tumor_model.joblib"
IMG_SIZE = 128
MAX_SAMPLES_PER_CLASS = None   # set small int to shorten run (e.g. 200) or None for all
PROGRESS_EVERY = 100           # print progress every N images
# ----------------------------

def load_cnn_extractor():
    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    try:
        return tf.keras.Model(inputs=cnn.input, outputs=cnn.layers[-2].output)
    except (AttributeError, ValueError):
        inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
        x = inp
        for layer in cnn.layers[:-1]:
            x = layer(x)
        model = tf.keras.Model(inputs=inp, outputs=x)
        _ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32), verbose=0)
        return model

def preprocess_for_cnn(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")/255.0
    img = img[..., np.newaxis]
    return img[np.newaxis, ...]

def preprocess_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img = clahe.apply(img)
    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_morph_features(mask):
    if mask is None: return np.array([0.0,0.0,0.0], dtype=np.float32)
    lbl = label(mask)
    props = regionprops(lbl)
    if not props: return np.array([0.0,0.0,0.0], dtype=np.float32)
    r = props[0]
    area = float(r.area)
    border = binary_dilation(mask) ^ mask
    perimeter = float(np.count_nonzero(border))
    bbox = r.bbox
    if len(bbox) >= 4:
        h = bbox[2] - bbox[0]; w = bbox[3] - bbox[1]
        bbox_vol = float(h * w)
    else:
        bbox_vol = float(area * 4)
    return np.array([area, perimeter, bbox_vol], dtype=np.float32)

def collect_image_paths(root_dir):
    imgs, labels = [], []
    notumor = os.path.join(root_dir, "notumor")
    tumor = os.path.join(root_dir, "tumor")
    if os.path.isdir(notumor):
        files = sorted([f for f in os.listdir(notumor) if os.path.isfile(os.path.join(notumor,f))])
        if MAX_SAMPLES_PER_CLASS: files = files[:MAX_SAMPLES_PER_CLASS]
        for fn in files:
            imgs.append(os.path.join(notumor, fn)); labels.append(0)
    if os.path.isdir(tumor):
        for sub in sorted(os.listdir(tumor)):
            subp = os.path.join(tumor, sub)
            if not os.path.isdir(subp): continue
            files = sorted([f for f in os.listdir(subp) if os.path.isfile(os.path.join(subp,f))])
            if MAX_SAMPLES_PER_CLASS: files = files[:MAX_SAMPLES_PER_CLASS]
            for fn in files:
                imgs.append(os.path.join(subp, fn)); labels.append(1)
    return imgs, labels

def main():
    t0 = time.time()
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    print("Loading CNN and building extractor...")
    extractor = load_cnn_extractor()
    print("Collecting image paths...")
    image_paths, labels = collect_image_paths(ROOT_DIR)
    if len(image_paths) == 0:
        raise RuntimeError("No images found. Check ROOT_DIR and its subfolders.")
    print(f"Collected {len(image_paths)} images ({sum(labels)} positive).")
    X, y = [], []
    for i, p in enumerate(image_paths, 1):
        cnn_in = preprocess_for_cnn(p)
        mask = preprocess_mask(p)
        if cnn_in is None or mask is None:
            # skip unreadable
            continue
        deep = extractor.predict(cnn_in, verbose=0)[0]
        morph = extract_morph_features(mask)
        comb = np.concatenate([deep, morph]).astype(np.float32)
        X.append(comb); y.append(labels[i-1])
        if i % PROGRESS_EVERY == 0:
            print(f"Processed {i}/{len(image_paths)}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("Feature matrix shape:", X.shape)
    if X.size == 0:
        raise RuntimeError("No features extracted; aborting.")
    print("Training RandomForest on combined features...")
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42)
    clf.fit(X, y)
    print("Saving RF to:", RF_SAVE_PATH)
    joblib.dump(clf, RF_SAVE_PATH)
    print("Done. Total time: %.1f s" % (time.time()-t0))
    # quick eval on same data
    pred = clf.predict(X)
    print(classification_report(y, pred))
    print("Confusion matrix:\n", confusion_matrix(y, pred))

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    main()
