import os
import cv2
import numpy as np
import nibabel as nib
from skimage.measure import regionprops, label
from scipy.ndimage import binary_dilation
from multiprocessing import Pool, cpu_count, freeze_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Paths ===
root_dir = r"D:\GBM_code\archive\Training"  # Kaggle dataset path
brats_dir = r"D:\GBM_code\GBM datasets\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"        # BRATS dataset path

IMG_SIZE = 128
MAX_SAMPLES = 100  # Limit samples for test speed

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
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    _, binary = cv2.threshold(img, 30, 1, cv2.THRESH_BINARY)
    return binary

def process_kaggle_image(args):
    img_path, label_val, tumor_type = args
    mask = preprocess_img(img_path)
    if mask is None:
        return None
    feats = extract_morph_features(mask) if label_val == 1 else [0, 0, 0]
    return (feats, label_val, tumor_type, "kaggle")

def process_brats_folder(subj_path):
    print(f"Processing BRATS subject folder: {subj_path}")
    mask_files = [f for f in os.listdir(subj_path) if f.endswith('seg.nii.gz')]
    results = []
    for mask_file in mask_files:
        mask_path = os.path.join(subj_path, mask_file)
        try:
            mask = nib.load(mask_path).get_fdata()
            binary = (mask > 0).astype(np.uint8)
            feats = extract_morph_features(binary)
            results.append((feats, 1, "gbm", "brats"))
            print(f"  Processed {mask_path}")
        except Exception as e:
            print(f"  Failed to load/process {mask_path}: {e}")
            continue
    print(f"Completed BRATS folder: {subj_path}")
    return results

if __name__ == "__main__":
    freeze_support()  # Needed on Windows for multiprocessing

    kaggle_args = []

    # NOTUMOR (limit 100)
    notumor_dir = os.path.join(root_dir, "notumor")
    for img_name in os.listdir(notumor_dir)[:MAX_SAMPLES]:
        kaggle_args.append((os.path.join(notumor_dir, img_name), 0, 'none'))

    # TUMOR classes (limit 100 per class)
    tumor_dir = os.path.join(root_dir, "tumor")
    for class_folder in os.listdir(tumor_dir):
        class_path = os.path.join(tumor_dir, class_folder)
        label_val = 1 if class_folder.lower() == 'glioma' else 0
        for img_name in os.listdir(class_path)[:MAX_SAMPLES]:
            kaggle_args.append((os.path.join(class_path, img_name), label_val, class_folder.lower()))

    print("Starting parallel processing for Kaggle images...")
    with Pool(cpu_count()) as pool:
        kaggle_results = pool.map(process_kaggle_image, kaggle_args)
    kaggle_results = [r for r in kaggle_results if r is not None]

    # BRATS subjects folder list limited to 100
    subj_folders = [
        os.path.join(brats_dir, sf)
        for sf in os.listdir(brats_dir)
        if os.path.isdir(os.path.join(brats_dir, sf))
    ][:MAX_SAMPLES]

    print("Starting parallel processing for BRATS subjects...")
    with Pool(cpu_count()) as pool:
        brats_lists = pool.map(process_brats_folder, subj_folders)

    # Flatten BRATS results list of lists
    brats_results = [item for sublist in brats_lists for item in sublist]

    # Combine all results
    all_results = kaggle_results + brats_results

    # Validate feature vector lengths
    for i, feat in enumerate(all_results):
        if len(feat[0]) != 3:
            print(f"Feature length mismatch at index {i}: {feat[0]}")

    features = [r[0] for r in all_results]
    labels = [r[1] for r in all_results]
    tumor_types = [r[2] for r in all_results]
    sources = [r[3] for r in all_results]

    X = np.array(features)
    y = np.array(labels)

    print(f"Total samples for classification: {len(X)}")

    # Train/test split and classification
    X_train, X_test, y_train, y_test, tt_train, tt_test, src_train, src_test = train_test_split(
        X, y, tumor_types, sources, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    for i, p in enumerate(pred):
        src = src_test[i]
        if p == 1:
            print(f"Sample {i} ({src}): yes (GBM/glioma detected)")
        else:
            ttype = tt_test[i]
            if ttype == 'none':
                print(f"Sample {i} ({src}): no (no tumor)")
            else:
                print(f"Sample {i} ({src}): no ({ttype} tumor)")

    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
