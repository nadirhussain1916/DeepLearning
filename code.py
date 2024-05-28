import numpy as np
import h5py
import os
from PIL import Image

# Function to resize and convert images to numpy arrays
def process_images(folder_path, image_size):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img = Image.open(os.path.join(folder_path, filename))
            img = img.resize((image_size, image_size))
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)

# Load and preprocess training images
training_images = process_images("training_set", image_size=64)

# Load and preprocess testing images
testing_images = process_images("testing_set", image_size=64)

# Convert images to HDF5 format
def convert_to_hdf5(images, labels, output_file):
    with h5py.File(output_file, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("labels", data=labels)

# Assuming labels are binary (0 for non-cat, 1 for cat)
# You need to replace these labels with your actual labels
# Also, make sure the labels are in the same order as the images
training_labels = np.random.randint(2, size=len(training_images))
testing_labels = np.random.randint(2, size=len(testing_images))

convert_to_hdf5(training_images, training_labels, "training_data.h5")
convert_to_hdf5(testing_images, testing_labels, "testing_data.h5")

# Load dataset from HDF5 files
def load_dataset():
    with h5py.File("training_data.h5", "r") as train_file, h5py.File("testing_data.h5", "r") as test_file:
        train_set_x_orig = np.array(train_file["images"])
        train_set_y_orig = np.array(train_file["labels"])
        test_set_x_orig = np.array(test_file["images"])
        test_set_y_orig = np.array(test_file["labels"])
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

# Define single-layer neural network model
class NeuralNetwork:
    def __init__(self, lambda_reg=0.01):
        self.weights = None
        self.bias = None
        self.lambda_reg = lambda_reg

    def train(self, X, y, learning_rate=0.001, num_iterations=1000):
        num_samples = X.shape[0]
        num_features = np.prod(X.shape[1:])
        X_flatten = X.reshape(num_samples, num_features)
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(num_iterations):
            y_pred = self.predict(X_flatten)
            dw = (1 / num_samples) * np.dot(X_flatten.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Regularization term gradient
            dw_reg = self.lambda_reg * self.weights

            # Update gradients with regularization
            dw += dw_reg

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        num_samples = X.shape[0]
        return np.where(np.dot(X.reshape(num_samples, -1), self.weights) + self.bias > 0, 1, 0)

# Train the model
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = load_dataset()
model = NeuralNetwork(lambda_reg=0.01)  # You can adjust lambda_reg as needed
model.train(train_set_x_orig, train_set_y_orig)

# Evaluate the model
train_predictions = model.predict(train_set_x_orig)
test_predictions = model.predict(test_set_x_orig)
train_accuracy = np.mean(train_predictions == train_set_y_orig)
test_accuracy = np.mean(test_predictions == test_set_y_orig)

# Print the results
print("Number of images in training dataset:", len(train_set_x_orig))
print("Number of images in testing dataset:", len(test_set_x_orig))
print("Number of cat images in training dataset:", np.sum(train_set_y_orig))
print("Number of non-cat images in training dataset:", len(train_set_y_orig) - np.sum(train_set_y_orig))
print("Number of cat images in testing dataset:", np.sum(test_set_y_orig))
print("Number of non-cat images in testing dataset:", len(test_set_y_orig) - np.sum(test_set_y_orig))
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

# Function to test a single image
# def test_single_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize((64, 64))
#     img_array = np.array(img)
#     prediction = model.predict(img_array.flatten())
#     if prediction == 1:
#         print("The image is a cat.")
#     else:
#         print("The image is not a cat.")
#
# # Example usage to test a single image
# test_single_image("test_image.jpg")
