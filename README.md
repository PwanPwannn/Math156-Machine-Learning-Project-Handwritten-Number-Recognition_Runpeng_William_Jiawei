
# CNN for MNIST Handwritten Digit Recognition

## Overview
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The code is written in Python using TensorFlow and Keras and is designed to train, evaluate, and visualize the model's performance on handwritten digit classification. Refer to the Project code with the "Latest" version.

## Project Directory Structure
Ensure that your project folder is organized as follows:

```
Math156_Final_Project/
├── data/
│   ├── mnist_train.csv
│   ├── mnist_test.csv
├── Math_156_Final_Project_Code_Latest.py
├── README.md
├── requirements.txt
```

- `data/`: Contains the dataset files (`mnist_train.csv` and `mnist_test.csv`).
- `Math_156_Final_Project_Code.py`: The main script for training and evaluating the model.
- `README.md`: Project documentation (this file).
- `requirements.txt`: List of dependencies required to run the project.

## Prerequisites
To run this project, you need the following:

- Python 3.8 or above (compatible with TensorFlow)
- Anaconda (recommended for managing virtual environments)

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository_url>
   cd Math156_Final_Project
   ```

2. **Create a Virtual Environment**
   - Use Anaconda to create a virtual environment:
     ```
     conda create -n tensorflow_env python=3.8
     conda activate tensorflow_env
     ```

3. **Install Dependencies**
   - Install the required packages using `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

4. **Prepare Dataset**
   - Ensure that the `data` folder exists in the project directory and contains `mnist_train.csv` and `mnist_test.csv` files.

5. **Run the Code**
   - Run the main script to train and evaluate the model:
     ```
     python Math_156_Final_Project_Code.py
     ```

## Code Description

1. **Data Loading and Preprocessing**
   - The `load_and_preprocess_data(filepath)` function is used to load the CSV files and normalize the images to have values between 0 and 1.
   - The labels are one-hot encoded.

2. **Data Augmentation**
   - The `augment_data(train_images, train_labels)` function is used to perform data augmentation, including rotations, shifts, shear, and zoom operations to improve model generalization.

3. **Model Definition**
   - The `create_cnn_model()` function defines a CNN model using TensorFlow and Keras. The model has multiple convolutional layers, batch normalization, and dropout for regularization.

4. **Model Training**
   - The `compile_and_train_model(model, datagen, train_images, train_labels)` function compiles and trains the model using the Adam optimizer and early stopping to avoid overfitting.

5. **Model Evaluation**
   - The `evaluate_model(model, test_images, test_labels)` function evaluates the model's performance on the test set and prints the accuracy, classification report, and confusion matrix.

6. **Plotting Training History**
   - The `plot_history(history)` function plots the training and validation accuracy and loss over epochs.

## Results
- The training and validation performance metrics are printed, and accuracy and loss plots are generated after training.
- The test set accuracy is displayed, along with a detailed classification report and confusion matrix to analyze model performance.

## Dependencies
The project requires the following Python packages:

- `tensorflow==2.11.0`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`

These dependencies are listed in `requirements.txt` for easy installation.

## Notes
- The current script uses relative paths (`data/mnist_train.csv`, `data/mnist_test.csv`) to ensure that it can be run on any machine with the correct folder structure.
- Make sure to keep the `data` folder in the same directory as the script.
