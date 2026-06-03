# Face Detection and Recognition using Haar Cascades and LBPH

This project implements a complete face detection and recognition system using classical Computer Vision techniques available in OpenCV.

The system combines Haar Cascade classifiers for object detection and Local Binary Pattern Histograms (LBPH) for face recognition, allowing real-time identification of registered users through a webcam.

The project was developed as part of a Computer Vision course and demonstrates the complete workflow from dataset creation to real-time facial recognition.

## Project Objectives

* Detect faces in real time.
* Detect eyes, smiles, and upper body regions.
* Build a facial image dataset.
* Train a face recognition model.
* Recognize registered individuals through a webcam.
* Analyze the performance of Haar Cascade classifiers.

## System Workflow

The project is divided into three main stages:

```text
Webcam Capture
       ↓
Face Detection (Haar Cascade)
       ↓
Dataset Creation
       ↓
LBPH Training
       ↓
Model Generation
       ↓
Real-Time Recognition
```

## Stage 1: Dataset Creation

Before recognition can be performed, the system must learn facial characteristics from training images.

### Data Acquisition

The webcam is used to capture facial images.

For each captured image:

* Face is detected automatically.
* Region of Interest (ROI) is extracted.
* Converted to grayscale.
* Resized to 100 × 100 pixels.
* Stored in a folder corresponding to the person's name.

### Dataset Structure

```text
dataset/
│
├── Rafael/
├── Daniela/
├── Ulises/
├── ...
```

Each folder contains multiple facial samples of a single individual.

## Stage 2: Face Recognition Training

Once the dataset is created, the recognition model is trained.

### LBPH Face Recognizer

The project uses:

```python
cv2.face.LBPHFaceRecognizer_create()
```

The algorithm:

1. Reads all training images.
2. Assigns a numerical ID to each person.
3. Extracts facial texture descriptors.
4. Trains the recognition model.
5. Stores the trained model.

Generated files:

* modelo.xml
* etiquetas.txt

## How LBPH Works

LBPH (Local Binary Pattern Histograms) describes facial textures by comparing each pixel with its neighbors.

For each region of the face:

1. Binary texture patterns are generated.
2. Histograms are computed.
3. Histograms are concatenated into a facial descriptor.

Recognition is performed by comparing descriptors from new images against the stored database.

### Advantages

* Simple implementation.
* Low computational requirements.
* Works well on small datasets.
* Suitable for real-time applications.

## Stage 3: Real-Time Detection and Recognition

The trained model is applied to live webcam frames.

### Preprocessing

Each frame undergoes:

* Grayscale conversion.
* Histogram equalization.
* Object detection.

### Supported Detectors

The system includes multiple Haar Cascade models:

#### Face Detection

* Frontal face
* Profile face

#### Facial Features

* Eyes
* Eyes with glasses
* Smile

#### Body Detection

* Upper body
* Full body

### Recognition Process

```text
Detected Face
      ↓
ROI Extraction
      ↓
LBPH Prediction
      ↓
Identity + Confidence
```

The recognizer returns:

* Predicted person.
* Similarity score.
* Recognition confidence.

## Haar Cascade Classifiers

The project uses the classical Viola-Jones framework.

### Integral Image

The integral image accelerates feature computation by storing cumulative pixel sums.

Benefits:

* Fast region evaluation.
* Real-time processing.
* Efficient feature extraction.

### Cascade Structure

Instead of analyzing every image region in detail:

1. Simple filters reject non-face regions.
2. More complex stages analyze remaining candidates.
3. Only promising regions continue through the pipeline.

Advantages:

* Significant speed improvement.
* Reduced computational cost.
* Real-time execution.

## Detection Parameters

Different classifiers were configured with optimized parameters.

| Detector          | scaleFactor | minNeighbors |
| ----------------- | ----------- | ------------ |
| Frontal Face      | 1.15        | 6            |
| Profile Face      | 1.15        | 6            |
| Upper Body        | 1.15        | 4            |
| Full Body         | 1.10        | 4            |
| Eyes              | 1.10        | 8            |
| Eyes with Glasses | 1.10        | 8            |
| Smile             | 1.20        | 20           |

## Results

The system successfully achieved:

* Real-time face detection.
* Eye detection.
* Smile detection.
* Facial recognition of registered users.
* Dataset generation and model retraining.

### Observations

* Face and eye detection produced reliable results.
* Smile detection worked under controlled conditions.
* Body detection generated more false positives.
* Recognition accuracy depended heavily on lighting conditions and training image quality.

## Technologies

* Python
* OpenCV
* Haar Cascades
* LBPH
* NumPy
* Webcam Interface

## Applications

The techniques implemented in this project can be applied to:

* Access control systems.
* Attendance monitoring.
* Human-computer interaction.
* Smart surveillance.
* Educational computer vision projects.
* Identity verification systems.

## Limitations

Several limitations were observed:

* Sensitivity to illumination changes.
* Reduced performance with varying backgrounds.
* Limited robustness to pose variations.
* Body detectors produce frequent false positives.
* Small training datasets affect recognition accuracy.

## Future Improvements

* Replace Haar Cascades with deep learning detectors.
* Use FaceNet or ArcFace embeddings.
* Improve recognition under varying lighting conditions.
* Increase dataset diversity.
* Add anti-spoofing mechanisms.
* Deploy the system as a web application.

## Project Files

| File          | Description           |
| ------------- | --------------------- |
| Capturar.py   | Dataset acquisition   |
| Train_LBPH.py | Model training        |
| Test.py       | Real-time recognition |
| modelo.xml    | Trained model         |
| etiquetas.txt | Label mapping         |
| dataset/      | Training images       |

## Author

Universidad de Guanajuato – Computer Vision

Rafael Alejandro Frías Cortez
