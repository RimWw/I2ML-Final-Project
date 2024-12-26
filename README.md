# Final Project: [Medical Image Classification Focusing on Pneumonia for Children-Chest part using VGG16]  

## A. Dataset  
We used the Pediatric Pneumonia Chest X-Ray dataset, available at:  
[https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)  

The dataset is organized into two folders:  
- **`train_dataset`**: Contains the training data.  
- **`test_dataset`**: Contains the testing data.  

## B. Code & Models  

### 1. Pre-trained Model  
- **`vgg16-397923af.pth`**:  
  This is the pre-trained VGG16 model we are improving, modified specifically for pediatric cases. Since it somehow can't be put inside the repository. We provide a Google Drive link directing to the model instead: 
  https://drive.google.com/file/d/16bvVWVtCP3OwHGf4EL4OPbpxWFX1yDVF/view?usp=sharing 

### 2. Training Script  
- **`Final_Version.ipynb`**:  
  A Jupyter notebook used to fine-tune the base model for the pediatric age group.  

### 3. Trained Model  
- **`best-weighted.pt`**:  
  The optimized model produced after training.  

## C. UI and Inference  

### 1. User Interface  
- **`ui.py`**:  
  Run this script to launch the user interface in a web browser.  

### 2. Prediction and Visualization  
- **`model.py`**:  
  This script loads the trained model (`best-weighted.pt`) to:  
  - Predict outcomes on the dataset.  
  - Generate Grad-CAM visualizations for better interpretability.  
