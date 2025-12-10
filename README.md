# GBM HYBRID DETECTION — SHORT README

## HOW IT WORKS
This project detects tumor vs non-tumor MRI slices using a hybrid model:
1. A CNN is trained on MRI images to learn deep features.
2. A preprocessing pipeline creates a binary mask of the brain region using CLAHE, thresholding, and morphological operations.
3. Morphological descriptors (area, perimeter, bounding-box volume) are extracted from this mask.
4. Deep CNN features + morphological features are combined into one feature vector.
5. A RandomForest classifier is trained on these combined features.
6. During detection, the same hybrid feature vector is extracted from a new image and fed to the RandomForest to output tumor probability + label.

Included scripts:
- **cnn_train.py** — trains the CNN.
- **train_rf_on_combined.py** — extracts features and trains the RandomForest.
- **combined_detect.py** — performs final prediction on a single image.

Excluded:
- cnn_detect.py, gbm_yn.py (not used in the hybrid pipeline).

---

## LIBRARIES USED
- **tensorflow / keras** — CNN training + feature extraction  
- **opencv-python (cv2)** — image loading, preprocessing, masks  
- **scikit-image** — morphology operations  
- **numpy** — numerical operations  
- **scikit-learn** — RandomForest classifier, metrics  
- **joblib** — saving/loading RF models  
- **scipy** — additional image processing utilities  

---

## IMPLEMENTATION SUMMARY

### 1. CNN Training (cnn_train.py)
- Loads dataset structured as:
  DATA_ROOT/notumor/…  
  DATA_ROOT/tumor/...  
- Applies augmentation and trains a small ConvNet.
- Saves model as `.keras` and `.h5`.

### 2. Feature Extraction + RF Training (train_rf_on_combined.py)
- Loads the trained CNN and converts it to a feature-extractor model.
- For every image:
  - Preprocess for CNN → get deep features  
  - Preprocess for mask → extract area, perimeter, bbox  
  - Concatenate both feature types  
- Trains a RandomForest on these combined features.
- Saves the RF model as `.joblib`.

### 3. Detection Pipeline (combined_detect.py)
- Loads CNN feature extractor + trained RandomForest.
- Preprocesses the input image the same way as in training.
- Extracts hybrid feature vector.
- Predicts tumor probability and final label.
- Can save debug visualizations (mask, overlay).

---

## OUTPUT 
1. GBM <img width="422" height="67" alt="image" src="https://github.com/user-attachments/assets/e3a2c3b6-e3e1-4ae8-b75e-f39268cddea2" />
    ![gbm4](https://github.com/user-attachments/assets/786efa3b-6113-4a91-9fef-fc4c08b497a1)

2. meningeoma <img width="391" height="68" alt="image" src="https://github.com/user-attachments/assets/2eeffb7d-17d1-4308-bbc5-013eebd990db" />
    ![menin](https://github.com/user-attachments/assets/4edb103c-3a32-4706-a458-67d1259575ec)

3. no tumor <img width="375" height="65" alt="image" src="https://github.com/user-attachments/assets/f0fc68c9-5015-44f4-81fb-ee507a823497" />
   ![nig](https://github.com/user-attachments/assets/7fca2353-e009-4e0b-af73-4f7ee06fe7b6)

4. gliosarcoma <img width="385" height="66" alt="image" src="https://github.com/user-attachments/assets/6fdc8e82-2208-4930-a215-65288467a096" />
   ![gs](https://github.com/user-attachments/assets/9b8a9e6c-7d3a-4dcd-a125-02a9e36e1ce5)






