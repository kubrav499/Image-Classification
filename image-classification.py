import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
training_images, testing_images = training_images / 255.0 , testing_images / 255.0

# Define class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Show a few images from the dataset
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap= plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate the model on testing data
loss, accuracy = model.evaluate(testing_images, testing_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save('image_classifier.model')

# Load the model
model = models.load_model('image_classifier.model')

# Load and preprocess the image of the horse
img = cv2.imread('images/horse.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0  # Normalize pixel values

# Show the image
plt.imshow(img, cmap= plt.cm.binary)
plt.show()

# Make prediction on the image
prediction = model.predict(np.array([img]))
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()

