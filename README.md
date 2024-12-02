# CNN for MNIST Handwritten Digit Recognition

## Overview
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The code is written in Python using TensorFlow and Keras and is designed to train, evaluate, and visualize the model's performance on handwritten digit classification. The project also includes a web-based application built using Streamlit that allows users to upload images of handwritten digits and see predictions in real-time.

## Project Directory Structure
Ensure that your project folder is organized as follows:

```
Math156_Final_Project/
├── data/
│   ├── mnist_train.csv
│   ├── mnist_test.csv
├── saved_model/
│   └── my_trained_cnn_model.h5
├── Math_156_Final_Project_Code_Latest.py  # Main script for training and evaluating the model
├── handwritten_digit_app.py  # Streamlit app for digit recognition
├── README.md  # Project documentation (this file)
├── requirements.txt  # List of dependencies required to run the project
```

- **data/**: Contains the dataset files (`mnist_train.csv` and `mnist_test.csv`).
- **saved_model/**: Contains the saved trained model (`my_trained_cnn_model.h5`).
- **Math_156_Final_Project_Code_Latest.py**: The main script for training and evaluating the model.
- **handwritten_digit_app.py**: The script for running the Streamlit app to predict handwritten digits.
- **README.md**: This documentation file.
- **requirements.txt**: File listing all the necessary dependencies.

## Prerequisites
To run this project, you'll need:
- Python 3.9 or later
- Anaconda or a similar environment manager (recommended)

### Setting Up the Environment
1. **Create and activate a new virtual environment** using Anaconda:
   ```sh
   conda create -n tensorflow_env python=3.9
   conda activate tensorflow_env
   ```

2. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Training Script
1. Make sure the **MNIST dataset** files are located in the `data` folder.
2. Run the training script to train the model and save it:
   ```sh
   python Math_156_Final_Project_Code_Latest.py
   ```
   The trained model will be saved in the `saved_model` folder as `my_trained_cnn_model.h5`.

## Running the Streamlit Application
The Streamlit application allows users to upload handwritten digit images and see predictions.

### Important Note on File Extensions
Make sure that all `.py` files have the correct `.py` extension and are not hidden or mislabeled (e.g., `.py.txt`). You can do this by enabling **file name extensions** in File Explorer:
1. Open **File Explorer**.
2. Click on the **View** tab.
3. Check the box labeled **File name extensions** to ensure you can see the correct file extensions.

### Running the Streamlit App (VERY IMPORTANT, BECAREFUL!)
Follow these steps to run the Streamlit application:

1. **Activate the Conda Environment**:
   ```sh
   conda activate tensorflow_env
   ```

2. **Navigate to the Project Directory**:
   ```sh
   cd "c:/Users/your_username/Desktop/Math156/Math156_Final_Project"
   ```

3. **Run the Streamlit App**:
   ```sh
   streamlit run handwritten_digit_app.py
   ```
   This command will open the app in your default web browser, where you can upload images and see the predictions.

## Dependencies
The necessary dependencies are listed in `requirements.txt`. If using Anaconda, you can install them all at once:
```sh
pip install -r requirements.txt
```

### Example `requirements.txt`
```
tensorflow==2.x.x
streamlit
Pillow
scikit-learn
numpy
```

Make sure to adjust TensorFlow's version (`2.x.x`) to the one compatible with your environment.

## Notes
- The model is saved in **HDF5 format** (`.h5`). This format works well with Keras, but you can also save it in the native Keras format (`.keras`) if preferred.
- Ensure your paths are correct and relative to the current working directory to allow seamless execution.

## Troubleshooting
- **Path Issues**: Ensure the `saved_model` and `data` folders are present and properly referenced.
- **Missing Dependencies**: Make sure to install all dependencies listed in `requirements.txt` before running the scripts.
- **File Extensions**: Ensure all Python files have the correct `.py` extension by enabling **file name extensions** in File Explorer.

