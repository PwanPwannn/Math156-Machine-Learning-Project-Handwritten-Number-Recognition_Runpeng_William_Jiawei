import tensorflow as tf
from tensorflow.keras import layers, models, utils
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    images = data[:, 1:].reshape(-1, 28, 28, 1).astype('float32') / 255
    labels = data[:, 0].astype('int')
    
    # One-hot encode the labels
    num_classes = 10
    labels_one_hot = utils.to_categorical(labels, num_classes)
    
    return images, labels_one_hot

# Define the CNN model using TensorFlow
def create_cnn_model():
    model = models.Sequential()
    
    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    # Third Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    
    # Flatten the output and add Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    
    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Compile and train the model
def compile_and_train_model(model, train_images, train_labels):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_images, train_labels, epochs=20, batch_size=64,
                        validation_split=0.2, callbacks=[early_stopping])
    
    return history

# Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')

# Plot the training history
def plot_history(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess training data
    train_data_path = r"C:\Users\runpengh1218\Desktop\Math156\Math156 Final Project\mnist_train.csv"
    test_data_path = r"C:\Users\runpengh1218\Desktop\Math156\Math156 Final Project\mnist_test.csv"
    
    if os.path.exists(train_data_path):
        train_images, train_labels = load_and_preprocess_data(train_data_path)
        
        # Create the CNN model
        cnn_model = create_cnn_model()
        
        # Compile and train the model
        history = compile_and_train_model(cnn_model, train_images, train_labels)
        
        # Plot training history
        plot_history(history)
        
        # Load and preprocess test data
        if os.path.exists(test_data_path):
            test_images, test_labels = load_and_preprocess_data(test_data_path)
            
            # Evaluate the model on the test set
            evaluate_model(cnn_model, test_images, test_labels)
        else:
            print("Dataset file 'mnist_test.csv' not found. Please provide the test dataset.")
    else:
        print("Dataset file 'mnist_train.csv' not found. Please provide the training dataset.")

