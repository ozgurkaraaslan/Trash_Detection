# Import necessary libraries
import os, json
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set the directory where the image data is stored
data_dir = "./data"
print(os.listdir(data_dir))  # Print the contents of the data directory

# Handling paths to data directories
data_directory = Path(data_dir)
subdirectories = [d for d in data_directory.glob("**/") if d.is_dir()]


# Gather information about data in each subdirectory
data_info = []
for directory in subdirectories:
    dir_name = directory.name
    files_count = len(list(directory.glob("*.*")))
    if dir_name != data_dir:
        data_info.append({"Directory": dir_name, "Number of Files": files_count})

# Create a DataFrame to view data information
df = pd.DataFrame(data_info)
df = df.set_index("Directory")

print(df, "\n")
print(f"Total number of images: {df['Number of Files'].sum()},\n")

# Setting parameters for image processing
batch_sz = 30
img_size = (180, 180)
val_split = 0.15

# Generating augmented image data for training
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale the pixel values
    validation_split=val_split,  # Set validation split
    # Below are various transformations for augmentation
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.5,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=50,
    shear_range=0.3,
    fill_mode="nearest",
)

# Image data generator for validation set (no augmentation, just rescaling)
val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=val_split)

# Creating training and validation datasets
train_dataset = train_gen.flow_from_directory(
    data_dir,
    subset="training",
    seed=133,
    target_size=img_size,
    batch_size=batch_sz,
    class_mode="categorical",
    color_mode="rgb",
)

val_dataset = val_gen.flow_from_directory(
    data_dir,
    subset="validation",
    seed=133,
    target_size=img_size,
    batch_size=batch_sz,
    class_mode="categorical",
    color_mode="rgb",
)

# Constructing the model architecture
model = Sequential(
    [
        keras.layers.ZeroPadding2D(
            padding=(1, 1), input_shape=(img_size[0], img_size[1], 3)
        ),
        Conv2D(32, (3, 3), activation="relu"),
        Dropout(0.3),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        Dropout(0.3),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        Dropout(0.4),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(6, activation="softmax"),
    ]
)

# Visualize the model architecture
keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

# Setting up callbacks for early stopping, learning rate reduction, and model checkpointing
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, min_lr=0.01)
model_save = ModelCheckpoint(
    "trash_classifier.h5", monitor="val_loss", mode="min", save_best_only=True
)

# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

# Train the model with the training and validation datasets
training_history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_dataset),
    epochs=1,
    validation_data=val_dataset,
    validation_steps=len(val_dataset),
    callbacks=[early_stop, reduce_lr, model_save],
)

# After train_dataset is created
class_indices = train_dataset.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

# Evaluate the model and print the highest validation accuracy achieved
best_validation_score = max(training_history.history["val_categorical_accuracy"])
print(f"Highest Validation Accuracy: {best_validation_score}")

# Accuracy plots
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(training_history.history["categorical_accuracy"]) + 1),
    training_history.history["categorical_accuracy"],
    "b",
    label="Training Accuracy",
)
plt.plot(
    range(1, len(training_history.history["val_categorical_accuracy"]) + 1),
    training_history.history["val_categorical_accuracy"],
    "r",
    label="Validation Accuracy",
)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Class labels
class_labels = train_dataset.class_indices
print(class_labels)

# Predictions
predictions = model.predict(val_dataset)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_dataset.classes
class_labels = list(val_dataset.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="g", vmin=0, cmap="Blues", cbar=False)
plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=90)
plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
