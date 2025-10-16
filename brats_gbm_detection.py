import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

root_folder = r"D:\GBM_code\GBM datasets\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

print("Root folder:", root_folder)
subject_ids = os.listdir(root_folder)
print("Subject IDs:", subject_ids)

X = []
y = []

for subject in subject_ids:
    folder = os.path.join(root_folder, subject)
    print("\nChecking folder:", folder)
    if not os.path.isdir(folder):
        print("Not a directory, skipping.")
        continue
    files = os.listdir(folder)
    print("Files:", files)
    flair_file = [f for f in files if "t1c.nii.gz" in f]  # or "t2w.nii.gz"
    seg_file = [f for f in files if "seg.nii.gz" in f]

    if not flair_file or not seg_file:
        print("Missing flair or seg file, skipping.")
        continue
    flair_path = os.path.join(folder, flair_file[0])
    seg_path = os.path.join(folder, seg_file[0])
    try:
        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()
    except Exception as e:
        print("Error loading files:", e)
        continue

    tumor_pixels = flair[seg > 0]
    background_pixels = flair[seg == 0]
    if tumor_pixels.size == 0:
        tumor_pixels = np.array([0])
    if background_pixels.size == 0:
        background_pixels = np.array([0])
    features = [
        np.mean(tumor_pixels), np.std(tumor_pixels),
        np.mean(background_pixels), np.std(background_pixels)
    ]
    label = 1 if np.sum(seg) > 0 else 0
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
print("\nTotal samples:", len(X))

if len(X) == 0:
    print("No samples found! Check your folder paths and file names.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
