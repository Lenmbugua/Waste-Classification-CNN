# Waste Classification with a Convolutional Neural Network (CNN)

Deep Learning practical assignment: building and evaluating a CNN for image classification.
**Author(s):** Hellen Mbugua & Ivy Kadzo
**Course:** Deep Learning | **Tools:** Python, Google Colab, TensorFlow/Keras

## Problem
Classify a photo of a waste item into one of six categories: cardboard, glass, metal, paper, plastic or trash. Automatic recognition of waste material could support sorting at the source, for example in smart bins or on sorting lines.

## Dataset
- Kaggle: *Garbage Classification* (uploaded by cchangcs). [https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification.]
- 2,527 images, 6 classes: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.
- Split (stratified): 1,768 train / 379 validation / 380 test.
- Preprocessing: resized to 128 x 128, pixel values divided by 255.

## Model
A CNN trained from scratch (no pretrained weights):
- Data augmentation (random flip, rotation, zoom; training only)
- 3 x [Conv2D (32, 64, 128 filters, 3x3, ReLU) + MaxPooling 2x2]
- Flatten, Dense(128, ReLU), Dropout(0.5), Dense(6, softmax)
- Adam optimizer, sparse categorical cross-entropy loss, accuracy metric, class weights for the imbalance
- 4,288,454 parameters; early stopping and learning-rate reduction

## Results (test set of 380 unseen images)
- **Test accuracy: 72.9%** (validation accuracy about 70%)
- Macro precision / recall / F1: 0.709 / 0.746 / 0.711

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cardboard | 0.852 | 0.867 | 0.860 |
| glass | 0.780 | 0.605 | 0.681 |
| metal | 0.611 | 0.710 | 0.657 |
| paper | 0.835 | 0.798 | 0.816 |
| plastic | 0.770 | 0.644 | 0.701 |
| trash | 0.405 | 0.850 | 0.548 |

On six new photos from the internet the model got 2 of 6 correct, often with very high confidence in the wrong class. See the notebook for the discussion.

## Files
- `waste_classification_cnn.ipynb`: the full Colab notebook (data preparation, model, training, evaluation, predictions) with outputs.
- `waste_cnn.keras`: the trained model.
- `slides/`: presentation slides.
- `report.pdf`: short report.

## How to run
1. Open the notebook in Google Colab and set the runtime to a T4 GPU.
2. Run all cells in order. The dataset downloads automatically from Kaggle through `kagglehub`.
3. For the prediction demo, put your own photos in a Google Drive folder and name them `<class>_<anything>.jpg` (for example `glass_jar.jpg`).

## Limitations
Small dataset, imbalanced classes (trash has only 137 images), and the training photos share one style (single item, plain background), so the model does not generalise well to differently styled photos.

## Acknowledgments
- Dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification.
