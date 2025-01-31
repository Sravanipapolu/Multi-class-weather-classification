# Multi-class-weather-classification
A deep learning project for classifying weather conditions into Cloudy, Rain, Shine, and Sunrise using CNNs. Models include a baseline and ResNet, with preprocessing steps like normalization, resizing, and class balancing. Evaluation is based on precision, recall, and F1-score. Implemented in TensorFlow, Keras, and PyTorch on Google Colab.
## 📂 Dataset
The dataset contains **1125 images** categorized into **Cloudy, Rain, Shine, and Sunrise**.

📥 **Download the Dataset**  
The dataset is publicly available on **Mendeley Data**.  
👉 [Click here to access the dataset](https://data.mendeley.com/datasets/4drtyfjtfy/1)

### 📌 Steps to Use:
1. **Download the dataset manually** from the link above.
2. Extract the contents into the `data/` folder.
3. Proceed with running the training scripts.

## 🏆 Models Used
We implemented and compared two deep learning models for weather classification:

1️⃣ **Custom CNN (Convolutional Neural Network)**  
   - Designed from scratch for image classification.
   - Includes convolutional, pooling, and fully connected layers.
   - Tuned hyperparameters for better generalization.

2️⃣ **ResNet (Residual Network)**  
   - Pretrained **ResNet-50** model used for transfer learning.
   - Feature extraction from pre-trained layers, followed by fine-tuning.
   - Helps in better accuracy with limited data.

## 🔄 Cross-Validation Strategy
To ensure model robustness, we applied **k-fold cross-validation**:
- **Stratified k-Fold Cross-Validation (k=5)** to maintain class balance.
- Helps evaluate performance across multiple training-validation splits.
- Prevents overfitting and ensures the model generalizes well.

## 📈 Model Training
Run the following Jupyter Notebooks:

1️⃣ **Data Preprocessing:** `Project_Preprocessing.ipynb`  
   - Normalization, resizing, and class balancing.

2️⃣ **Model Training & Evaluation:** `Project_Model.ipynb`  
   - Train both **Custom CNN** and **ResNet** models.
   - Apply **cross-validation** and evaluate performance.

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**
- **Cross-validation performance comparison**

### **Attribution Requirement:**  
If you use this dataset, you must provide proper credit by citing:  
Ajayi, Gbeniniyi (2018), **“Multi-class Weather Dataset for Image Classification”**,  
Mendeley Data, V1, DOI: [10.17632/4drtyfjtfy.1](https://data.mendeley.com/datasets/4drtyfjtfy/1).
