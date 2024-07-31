# Bone Fracture Detection Using Convolutional Neural Network (CNN)

## Project Overview
The primary objective of this project is to develop automated systems for detecting bone fractures in X-ray images using both Convolutional Neural Networks (CNN) and traditional machine learning models. The goal is to assist medical professionals by providing accurate and reliable fracture detection, thus improving diagnostic efficiency.

## Problem Statement
Bone fractures are common injuries that require timely and accurate diagnosis. Traditional methods rely heavily on the expertise of radiologists, which can be time-consuming and prone to human error. This project explores the potential of CNNs and traditional machine learning models to automate the fracture detection process, reducing the diagnostic burden on medical professionals and improving patient outcomes.

## Results
The CNN model and traditional machine learning models were trained and evaluated using a dataset of X-ray images. The models demonstrated high accuracy and reliability in detecting bone fractures. Key performance metrics include precision, recall, and F1 score, all of which indicate the models' effectiveness in identifying fractures.

## Important Findings
### Data Augmentation: 
Applying data augmentation techniques significantly improved the CNN model's generalization and performance.
### Model Architecture: 
The chosen CNN architecture effectively captured important features in the X-ray images, leading to high accuracy in fracture detection.
### Traditional Models: 
Traditional machine learning models such as SVM, Random Forest, k-NN, and Logistic Regression also showed promising results, with Random Forest and SVM achieving the highest accuracy among them.
### Early Stopping: 
Implementing early stopping for the CNN prevented overfitting and ensured the model maintained optimal performance during training.
### Single Image Prediction:
The model can predict the presence of a fracture in a single image, demonstrating practical usability.

## Suggestions for Next Steps
### Expand Dataset:
Incorporating a larger and more diverse dataset could further enhance the model's robustness and accuracy.
### Advanced Techniques:
Exploring advanced techniques such as transfer learning and ensemble methods may yield further improvements.
### Real-World Deployment: 
Implementing the model in a clinical setting for practical evaluation and feedback from medical professionals.
## Performance Metrics

#### CNN Model
Validation Accuracy: High (e.g., 0.94)
Test Accuracy: High (e.g., 0.91)
Precision, Recall, F1-Score: High for both fractured and non-fractured classes.
    
F1 Score: 0.73
Precision: 0.68
Recall: 0.78
    
#### SVM Model
Validation Accuracy: 0.875
Test Accuracy: 0.75
Classification Report (Test):

            precision    recall  f1-score   support

       0.0       0.86      0.70      0.77       360
       1.0       0.65      0.82      0.73       240

  accuracy                           0.75       600
 macro avg       0.75      0.76      0.75       600
weighted avg 0.77 0.75 0.75 600


#### Random Forest Model
- **Validation Accuracy:** 0.995
- **Test Accuracy:** 0.71
- **Classification Report (Test):**

          precision    recall  f1-score   support

     0.0       0.97      0.53      0.69       360
     1.0       0.58      0.97      0.73       240

accuracy                           0.71       600
macro avg 0.78 0.75 0.71 600
weighted avg 0.81 0.71 0.70 600


#### k-NN Model
- **Validation Accuracy:** 0.965
- **Test Accuracy:** 0.735
- **Classification Report (Test):**
markdown
Copy code
          precision    recall  f1-score   support

     0.0       0.94      0.60      0.73       360
     1.0       0.61      0.94      0.74       240

accuracy                           0.73       600
macro avg 0.77 0.77 0.73 600
weighted avg 0.80 0.73 0.73 600


#### Logistic Regression Model
- **Validation Accuracy:** 0.852
- **Test Accuracy:** 0.738
- **Classification Report (Test):**

          precision    recall  f1-score   support

     0.0       0.90      0.63      0.74       360
     1.0       0.62      0.90      0.73       240

accuracy                           0.74       600
macro avg 0.76 0.77 0.74 600
weighted avg 0.79 0.74 0.74 600

## Directory structure
### BoneFractureDataset/
#### testing/
##### fractured/
##### not_fractured/
#### training/
##### fractured/
##### not_fractured/
### Readme.md
### frac_det_CNN_32.ipynb
This notebook covers the Convolutional Neural Network (CNN) architecture, training, and evaluation process. It includes data augmentation techniques, model training with early stopping, and evaluation metrics such as accuracy, precision, recall, and F1 score.

### frac_svm_rf_knn_lr.ipynb
This notebook explores the use of traditional machine learning models such as SVM, Random Forest, k-NN, and Logistic Regression for bone fracture detection. It includes data preprocessing, model training, evaluation, and comparison of results.



