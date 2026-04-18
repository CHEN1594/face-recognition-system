# Assignment 1 (Classical): Online Face Detection and Recognition

This folder contains a classical (non-deep-learning) pipeline for:
- face data collection
- image preprocessing
- model training (`PCA + LDA + KNN`)
- live camera recognition with temporal smoothing

## 1. Project Structure

- `data_collection.py`: collect cropped face images into `../gallery/<person_name>/`
- `preprocess.py`: preprocess gallery images and write to `../gallery_preprocessed/`
- `import_gallery_new.py`: import raw phone photos from `../gallery_new`, detect+preprocess, and append to `../gallery_preprocessed/<person>/` with continuous indices
- `train.py`: train and evaluate model, then save `../models/pca_lda_knn.joblib`
- `recognize_live.py`: live recognition with OpenCV Haar cascade + unknown threshold + time smoothing
- `haarcascade_frontalface_default.xml`: face detector file used by live recognition
- `environment.yml`: exported conda environment

## 2. Environment Setup

Run in PowerShell from repo root (`face-recognition-system`):

```powershell
conda env create -f assignment1-classical/environment.yml
conda activate frs
```

If you want manual install instead of `environment.yml`:

```powershell
conda create -n frs python=3.10 -y
conda activate frs
conda install -c conda-forge numpy opencv scikit-learn joblib matplotlib -y
```

## 3. Full Reproduction (Step by Step)

Go to this folder first:

```powershell
cd assignment1-classical
```

### Step A. (Optional) Collect face images

```powershell
python data_collection.py
```

Output goes to `../gallery/<name>/` as grayscale `90x90` face images.

### Step B. Preprocess gallery images

```powershell
python preprocess.py
```

Default input/output:
- input: `../gallery`
- output: `../gallery_preprocessed`

### Step B2. Import new raw photos (optional)

If you collected new raw photos in `../gallery_new` (not cropped), run:

```powershell
python import_gallery_new.py
```

This script will:
- detect face with Haar cascade
- crop + resize to `90x90`
- apply preprocessing (CLAHE + minmax normalization)
- append files into `../gallery_preprocessed/<person>/` as `<person>_<next_index>.jpg`

### Step C. Train model

```powershell
python train.py
```

Default output model:
- `../models/pca_lda_knn.joblib`

### Step D. Live recognition

```powershell
python recognize_live.py
```

Press `q` to quit.

## 4. If You Only Want Live Recognition

Minimum requirement:
- `../models/pca_lda_knn.joblib` already exists
- `haarcascade_frontalface_default.xml` exists in `assignment1-classical/`

Then run:

```powershell
cd assignment1-classical
conda activate frs
python recognize_live.py
```

## 5. Important Runtime Parameters (recognize_live.py)

Edit the top parameter block in `recognize_live.py`:

- `MODEL_PATH`: model file path
- `CAMERA_INDEX`: camera id (`0` for default webcam)
- `UNKNOWN_THRESHOLD`: larger value = easier to classify as known person
- `MIN_FACE_SIZE`, `SCALE_FACTOR`, `MIN_NEIGHBORS`: Haar detection sensitivity
- `SMOOTH_WINDOW`, `SMOOTH_MIN_COUNT`: temporal smoothing stability

Recommended tuning:
- if labels flicker: increase `SMOOTH_WINDOW` and `SMOOTH_MIN_COUNT`
- if response is too slow: decrease both smoothing values

## 6. Common Issues

- Error: cascade file not found / detector empty  
  Ensure `haarcascade_frontalface_default.xml` is in `assignment1-classical/`.

- Error: conda: command not found
  This means Conda is not installed on your system.
  Please install Conda (e.g., Miniconda or Anaconda) by following an online tutorial, then restart your terminal and try again.
