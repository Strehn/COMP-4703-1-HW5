"""
Homework Part 2: Train a CNN to Recognize CAPTCHA Letters

Goal: Build, train, and evaluate a convolutional neural network on extracted letter images.

Tasks:
- TODO: Parameterize hyperparameters via CLI (epochs, batch size) and justify choices.
- TODO: Experiment with regularization (e.g., Dropout) and report effects on overfitting.
- TODO: Save training/validation curves (accuracy/loss) and include in your write-up.
- TODO (optional): Try data augmentation and compare results.
"""

import argparse
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helpers import resize_to_fit


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Argument parsing for hyperparameters
parser = argparse.ArgumentParser(description="Train CAPTCHA letter recognition model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
args = parser.parse_args()

# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# TODO (optional): try standardization or augmentation here and compare.

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(32, activation="softmax"))

# TODO: Consider adding Dropout to reduce overfitting.

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch_size, epochs=args.epochs, verbose=1)

# Report final training metrics
final_train_acc = history.history.get("accuracy", [None])[-1]
final_val_acc = history.history.get("val_accuracy", [None])[-1]
print(f"[REPORT] Final training accuracy: {final_train_acc}")
print(f"[REPORT] Final validation accuracy: {final_val_acc}")

# Save the trained model to disk
model.save(MODEL_FILENAME)
