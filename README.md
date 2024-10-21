# Image Classification using Intel Image Dataset

## Project Overview
Developed a multi-class image classification system using the **Intel Image Classification dataset**, focusing on data exploration, custom CNN architectures, hyperparameter tuning, and transfer learning with MobileNet. The project showcases comparative evaluation to identify the most effective model.

---

## Table of Contents
1. [Data Exploration & Preprocessing](#data-exploration--preprocessing)
2. [Model Development](#model-development)
   - [Simple Classifier](#simple-classifier)
   - [Complex Classifier with Keras-Tuner](#complex-classifier-with-keras-tuner)
   - [Custom ConvNet Architecture](#convnet-architecture-implementation)
   - [Pre-trained Model - MobileNet](#pre-trained-model---mobilenet)
3. [Comparative Evaluation](#comparative-evaluation)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [How to Run the Code](#how-to-run-the-code)
7. [References](#references)

---

## 1. Data Exploration & Preprocessing
- Analyzed six image classes: **Street, Sea, Mountain, Glacier, Forest, and Buildings** to ensure balanced data distribution between training and test datasets.
- Verified **class proportions** between datasets to ensure consistency and avoid model overfitting.
- Loaded datasets using TensorFlow's `image_dataset_from_directory` function, preparing them for **training, validation, and testing**.

---

## 2. Model Development

### 2.1 Simple Classifier
- Developed a basic neural network with:
  - **Flatten layer** and appropriate **output layer** based on the number of classes.
  - Implemented **early stopping** using 20% of the training set for validation.
- Reported initial test accuracy and identified areas for improvement.

### 2.2 Complex Classifier with Keras-Tuner
- Tuned hyperparameters with **Keras-Tuner**, experimenting with:
  - **Number and sizes** of hidden layers.
  - **Dropout rates** for regularization.
  - **Learning rates** to ensure smooth convergence.

### 2.3 ConvNet Architecture Implementation
- Built a custom **CNN architecture** with:
  - **Four Conv2D layers** followed by **MaxPooling2D** for feature extraction.
  - Optimized hyperparameters like **filter sizes**, **kernel dimensions**, and **pool sizes** through experimentation.
- Achieved **54.67% test accuracy** with this custom architecture.

### 2.4 Pre-trained Model - MobileNet
- Used **MobileNet**, pre-trained on ImageNet, to perform **transfer learning**:
  - **Frozen MobileNet weights** during training and added a custom classification layer.
  - Trained the model on the Intel dataset and achieved **91.07% accuracy** on the test set.

---

## 3. Comparative Evaluation
- Conducted comparative analysis between the **custom CNN model** and the **MobileNet model**.
- Visualized results using a **confusion matrix** to identify misclassifications.
- Noted MobileNet’s **superior performance** with:
  - **MobileNet Accuracy:** 91.07%
  - **Custom CNN Accuracy:** 54.67%
- Displayed sample images with the **most common classification errors** for further analysis.

---

## 4. Results
- **Best Model:** MobileNet with 91.07% accuracy on the test set.
- **Common Misclassifications:** Errors primarily involved distinguishing between **similar categories**, such as **mountain vs. glacier** and **street vs. buildings**.

---

## 5. Conclusion
- The **MobileNet** model demonstrated significant improvement over the custom CNN, emphasizing the effectiveness of **transfer learning**.
- The project highlights the importance of **hyperparameter tuning**, **model selection**, and **data consistency** for achieving high accuracy in multi-class image classification tasks.

---

## 6. How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intel-image-classification.git
   cd intel-image-classification
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
## 7. Reference
- Intel Image Classification Dataset: Intel Open Data
- TensorFlow Documentation: TensorFlow
- MobileNet Paper: Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
