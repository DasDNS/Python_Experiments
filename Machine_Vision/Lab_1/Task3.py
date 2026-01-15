# ===============================
# TASK 3: CNN Facial Expression Detection
# ===============================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

# -------------------------------
# 1. DATASET PATHS
# -------------------------------
# Folder structure:
# Lab/
# └── data/
#     └── data/
#         ├── train/
#         └── test/

train_data_dir = r'data/data/train/'
validation_data_dir = r'data/data/test/'

# -------------------------------
# 2. DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# -------------------------------
# 3. DATA GENERATORS
# -------------------------------
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# -------------------------------
# 4. CLASS LABELS
# -------------------------------
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------------------
# 5. CNN MODEL ARCHITECTURE
# -------------------------------
model = Sequential()

# 1st Convolution Block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 2nd Convolution Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 3rd Convolution Block
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Output Layer (7 Classes)
model.add(Dense(7, activation='softmax'))

# -------------------------------
# 6. COMPILE MODEL
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# -------------------------------
# 7. COUNT TRAIN & TEST IMAGES
# -------------------------------
train_path = r'data/data/train/'
test_path = r'data/data/test/'

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print("Number of training images:", num_train_imgs)
print("Number of testing images:", num_test_imgs)

# -------------------------------
# 8. TRAIN THE MODEL
# -------------------------------
epochs = 3

history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32
)

# -------------------------------
# 9. SAVE THE MODEL
# -------------------------------
model.save('facial_expression_cnn_model.h5')

print("✅ Model training completed and saved successfully.")
