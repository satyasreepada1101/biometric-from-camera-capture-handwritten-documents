'''
running commands:
py -3.11 -m pip install --upgrade pip
py -3.11 alexnet_writer_identification.py

make sure to change the directory paths for the train and test datasets in the code before running it.
'''

import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Activation,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# DATASET PATHS
TRAIN_DIR = r"C:\Satya Sreepada\B.Tech\SEM 5\Machine Learning\End sem project\ML_dataset_patches\train"
TEST_DIR = r"C:\Satya Sreepada\B.Tech\SEM 5\Machine Learning\End sem project\ML_dataset_patches\test"

# PARAMETERS
IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 32
EPOCHS = 20

# COUNT WRITERS
num_classes = len([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

print("=" * 50)
print("Number of writers:", num_classes)
print("=" * 50)

# DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=5,

    width_shift_range=0.05,
    height_shift_range=0.05,

    zoom_range=0.10,

    horizontal_flip=False,

    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


print("\nClass Mapping:")
print(train_generator.class_indices)


# ALEXNET MODEL
model = Sequential()

# Layer 1
model.add(
    Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='same',
        input_shape=(224, 224, 3)
    )
)

model.add(Activation('relu'))

model.add(
    MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2)
    )
)

model.add(BatchNormalization())

# Layer 2
model.add(
    Conv2D(
        filters=256,
        kernel_size=(5, 5),
        padding='same'
    )
)

model.add(Activation('relu'))

model.add(
    MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2)
    )
)

model.add(BatchNormalization())

# Layer 3
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        padding='same'
    )
)

model.add(Activation('relu'))

# Layer 4
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        padding='same'
    )
)

model.add(Activation('relu'))

# Layer 5
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same'
    )
)

model.add(Activation('relu'))

model.add(
    MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2)
    )
)

# Flatten
model.add(Flatten())


# FC Layer 1
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# COMPILE
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# MODEL SUMMARY
model.summary()

# EARLY STOPPING
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# TRAINING
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop]
)


# EVALUATION
loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\n" + "=" * 50)
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print("=" * 50)

# PRINT AND SAVE ACTUAL/PREDICTED VALUES
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_class_ids = predictions.argmax(axis=1)
class_names = {
    class_id: class_name
    for class_name, class_id in test_generator.class_indices.items()
}

with open("writer_predictions.csv", "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Actual", "Predicted", "Confidence", "Correct"])

    for row_index, (actual_class_id, predicted_class_id) in enumerate(zip(
        test_generator.classes,
        predicted_class_ids
    )):
        actual_value = class_names[actual_class_id]
        predicted_value = class_names[predicted_class_id]
        confidence = predictions[row_index][predicted_class_id] * 100
        correct = "Yes" if actual_class_id == predicted_class_id else "No"
        print(
            f"Actual: {actual_value} | Predicted: {predicted_value} "
            f"| Confidence: {confidence:.2f}% | Correct: {correct}"
        )
        csv_writer.writerow([
            actual_value,
            predicted_value,
            f"{confidence:.2f}%",
            correct
        ])

print("\nPrediction values saved as:")
print("writer_predictions.csv")


# SAVE MODEL
model.save("alexnet_writer_identification.keras")

print("\nModel saved as:")
print("alexnet_writer_identification.keras")


# ACCURACY PLOT
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("AlexNet Writer Identification Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "alexnet_accuracy_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# LOSS PLOT
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("AlexNet Writer Identification Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "alexnet_loss_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()