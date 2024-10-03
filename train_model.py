import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define image size and batch size
img_size = 300
batch_size = 32

# Path to your dataset
train_data_path = r"D:\Programs\Sacred_Eye\project\Data"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values between 0 and 1
    rotation_range=20,        # Rotate images randomly
    width_shift_range=0.2,    # Shift the image horizontally
    height_shift_range=0.2,   # Shift the image vertically
    shear_range=0.2,          # Shearing for deformation
    zoom_range=0.2,           # Zoom in or out
    horizontal_flip=True,     # Flip horizontally
    fill_mode='nearest',      # Filling missing pixels after augmentation
    validation_split=0.2      # Use 20% of data for validation
)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'         # Training set
)

# Load and preprocess the validation dataset
validation_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'       # Validation set
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    
    Dense(train_generator.num_classes, activation='softmax')  # Number of output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,                # Adjust number of epochs based on your needs
    validation_data=validation_generator
)

# Save the model
model.save(r"D:\Programs\Sacred_Eye\project\Model\keras_model.h5")
