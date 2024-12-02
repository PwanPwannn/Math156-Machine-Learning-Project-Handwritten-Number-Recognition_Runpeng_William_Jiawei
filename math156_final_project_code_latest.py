import tensorflow as tf
from tensorflow.keras import layers, models, utils, preprocessing
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

# Data augmentation
def augment_data(train_images, train_labels):
    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    datagen.fit(train_images)
    return datagen

# Define the CNN model using TensorFlow
def create_cnn_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Compile and train the model
def compile_and_train_model(model, datagen, train_images, train_labels):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                        epochs=20, validation_data=(train_images, train_labels), callbacks=[early_stopping])
    
    return history

# Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')

    # Additional evaluation metrics
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))
    print("\nConfusion Matrix:\n", confusion_matrix(true_labels, predicted_labels))

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
    # Print the current working directory to verify the script's location
    print("Current working directory:", os.getcwd())
    # Set the working directory to the script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure the 'data' directory exists
    data_folder = "data"
    if not os.path.exists(data_folder):
        print(f"The '{data_folder}' directory does not exist. Please create a 'data' folder and place the dataset files in it.")
        exit()
    
    # Load and preprocess training data
    train_data_path = os.path.join(data_folder, "mnist_train.csv")
    test_data_path = os.path.join(data_folder, "mnist_test.csv")
    
    if os.path.exists(train_data_path):
        train_images, train_labels = load_and_preprocess_data(train_data_path)
        
        # Data augmentation
        datagen = augment_data(train_images, train_labels)
        
        # Create the CNN model
        cnn_model = create_cnn_model()
        
        # Compile and train the model
        history = compile_and_train_model(cnn_model, datagen, train_images, train_labels)
        
        # Plot training history
        plot_history(history)
        
        # Load and preprocess test data
        if os.path.exists(test_data_path):
            test_images, test_labels = load_and_preprocess_data(test_data_path)
            
            # Evaluate the model on the test set
            evaluate_model(cnn_model, test_images, test_labels)
        else:
            print("Dataset file 'mnist_test.csv' not found. Please provide the test dataset in the 'data' folder.")
    else:
        print("Dataset file 'mnist_train.csv' not found. Please provide the training dataset in the 'data' folder.")
        
cnn_model.save("saved_model/my_trained_cnn_model.h5")

