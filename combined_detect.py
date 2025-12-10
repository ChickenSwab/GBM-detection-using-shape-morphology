import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes, binary_dilation
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
IMG_SIZE = 128
IMAGE_PATH = r"C:\Users\chand\Downloads\gbm4.jpeg"

CNN_MODEL_PATH = r"D:\GBM_code_new\cnn_brain_tumor.h5"
RF_MODEL_PATH  = r"D:\GBM_code_new\rf_brain_tumor_model.joblib"

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
rf_model  = joblib.load(RF_MODEL_PATH)

# Fallback extractor
try:
    feature_extractor = tf.keras.Model(inputs=cnn_model.input,
                                       outputs=cnn_model.layers[-2].output)
except Exception:
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = inp
    for layer in cnn_model.layers[:-1]:
        x = layer(x)
    feature_extractor = tf.keras.Model(inputs=inp, outputs=x)
    _ = feature_extractor.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 1),
                                           dtype=np.float32), verbose=0)

# ------------------------------------------------------------
# CNN IMAGE PREPROCESS
# ------------------------------------------------------------
def preprocess_cnn_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------------------------------------
# ROBUST MORPHOLOGY BLOCK
# ------------------------------------------------------------
def _morph_cleanup(mask, min_size=40):
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    m = binary_fill_holes(m).astype(np.uint8)

    lbl = label(m)
    out = np.zeros_like(m)

    for prop in regionprops(lbl):
        if prop.area >= min_size:
            out[lbl == prop.label] = 1

    return out


def _largest_component(mask):
    lbl = label(mask)
    if lbl.max() == 0:
        return mask
    props = regionprops(lbl)
    biggest = max(props, key=lambda p: p.area)
    return (lbl == biggest.label).astype(np.uint8)


from skimage.morphology import remove_small_objects
import math

def _center_constraint(mask, keep_frac=0.6):
    """Keep regions whose centroid is within central box of image."""
    h, w = mask.shape
    ch0 = int((1 - keep_frac)/2 * h)
    ch1 = int(h - ch0)
    cw0 = int((1 - keep_frac)/2 * w)
    cw1 = int(w - cw0)
    central = np.zeros_like(mask)
    central[ch0:ch1, cw0:cw1] = 1
    return mask & central

def preprocess_mask_robust(path, save_debug=False, debug_prefix="debug"):
    """
    Improved robust mask:
     - CLAHE + median blur
     - Otsu, inverted Otsu, adaptive, kmeans candidates
     - remove small objects, morphological open/close, keep largest component
     - apply center-of-brain constraint to avoid skull
    Returns binary mask (0/1) same shape as IMG_SIZE x IMG_SIZE.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 1) Preproc
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    candidates = []

    # Otsu normal
    _, otsu = cv2.threshold(img_clahe, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("otsu", otsu))

    # Otsu inverted
    _, otsu_inv = cv2.threshold(255 - img_clahe, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("otsu_inv", otsu_inv))

    # Adaptive
    try:
        adp = cv2.adaptiveThreshold(img_clahe, 1,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 2)
        candidates.append(("adaptive", adp))
    except:
        pass

    # KMeans (2 clusters) on intensity
    try:
        pixels = img_clahe.reshape(-1, 1).astype(np.float32)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=0, n_init=4).fit(pixels)
        labels_km = km.labels_.reshape(img_clahe.shape)
        centers = km.cluster_centers_.flatten()
        # add both possible clusters
        for idx in [np.argmax(centers), np.argmin(centers)]:
            candidates.append((f"kmeans_{idx}", (labels_km == idx).astype(np.uint8)))
    except Exception:
        pass

    # morphological cleaning helper
    def clean_mask(m):
        if m is None: return np.zeros_like(img_clahe, dtype=np.uint8)
        m = m.astype(np.uint8)
        # open + close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        # fill holes
        m = binary_fill_holes(m).astype(np.uint8)
        # remove tiny objects
        m = remove_small_objects(m.astype(bool), min_size=40).astype(np.uint8)
        # ensure largest connected component only
        lbl = label(m)
        if lbl.max() == 0:
            return np.zeros_like(m)
        props = regionprops(lbl)
        biggest = max(props, key=lambda p: p.area)
        out = (lbl == biggest.label).astype(np.uint8)
        # center constraint to avoid skull picks
        out = _center_constraint(out, keep_frac=0.75).astype(np.uint8)
        # if center constraint removed everything, fallback to unconstrained
        if out.sum() == 0:
            out = (lbl == biggest.label).astype(np.uint8)
        return out

    # score candidate masks similarly to your original scoring
    best_mask, best_score, best_name = None, -1, None
    def score_mask(m):
        if m.sum() == 0: return 0
        lbl = label(m)
        props = regionprops(lbl)
        if not props: return 0
        p = props[0]
        area = p.area
        bbox = p.bbox
        if len(bbox) >= 4:
            h = bbox[2] - bbox[0]; w = bbox[3] - bbox[1]
            bbox_area = max(1, h*w)
        else:
            bbox_area = area
        solidity = area / bbox_area
        return (area**0.5) * solidity

    for name, m in candidates:
        cleaned = clean_mask(m)
        s = score_mask(cleaned)
        if s > best_score:
            best_score = s; best_mask = cleaned; best_name = name

    # fallback: aggressive dilation of otsu_inv then pick largest
    if best_mask is None or best_mask.sum() == 0:
        fallback = binary_dilation(otsu_inv, iterations=2).astype(np.uint8)
        best_mask = _largest_component(fallback)

    # save debug outputs (same naming convention you already use)
    if save_debug:
        orig = cv2.imread(path)
        if orig is not None:
            orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
            mask8 = (best_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = orig.copy()
            cv2.drawContours(overlay, contours, -1, (0,0,255), 2)
            blended = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
            cv2.imwrite(f"{debug_prefix}_overlay_{best_name}.png", blended)
            cv2.imwrite(f"{debug_prefix}_mask_{best_name}.png", mask8)

    return best_mask

def extract_morph_features_from_mask(mask):
    """
    Keep previous 3 features (area, perimeter, bbox_vol) so RF shape unchanged.
    But compute them more robustly.
    """
    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return np.array([0., 0., 0.], dtype=np.float32)
    r = props[0]
    area = float(r.area)

    # perimeter: more robust estimate
    border = binary_dilation(mask) ^ mask
    perimeter = float(np.count_nonzero(border))

    # bbox volume (2D area of bbox)
    bbox = r.bbox
    if len(bbox) >= 4:
        h = max(1, bbox[2] - bbox[0])
        w = max(1, bbox[3] - bbox[1])
        bbox_vol = float(h * w)
    else:
        bbox_vol = float(area * 4)

    # ensure no tiny/zero
    area = max(0.0, area)
    perimeter = max(0.0, perimeter)
    bbox_vol = max(0.0, bbox_vol)
    return np.array([area, perimeter, bbox_vol], dtype=np.float32)



# ------------------------------------------------------------
# BUILD HYBRID FEATURE VECTOR
# ------------------------------------------------------------
def extract_hybrid_features(path):
    cnn_input = preprocess_cnn_image(path)
    deep_feats = feature_extractor.predict(cnn_input, verbose=0)[0]

    mask = preprocess_mask_robust(path,
                                  save_debug=True,
                                  debug_prefix="detect_debug")
    morph = extract_morph_features_from_mask(mask)

    return np.concatenate([deep_feats, morph])


# ------------------------------------------------------------
# HYBRID PREDICT
# ------------------------------------------------------------
def hybrid_predict(path):
    print("\n🔍 Hybrid CNN + Morphology + RF Prediction")
    print("------------------------------------------")
    print("Image:", path)

    feats = extract_hybrid_features(path)
    prob = rf_model.predict_proba(feats.reshape(1, -1))[0][1]

    label = "Tumor" if prob >= 0.20 else "No Tumor"


    print(f"Prediction: {label}")
    print(f"Tumor Probability: {prob:.4f}")

    return label, prob


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    hybrid_predict(IMAGE_PATH)
