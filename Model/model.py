import tensorflow as tf 
import tensorflow_datasets as tfds
import math 
import numpy as np 
import matplotlib.pyplot as plt
import logging 
import cv2 as cv

datasets, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True) # Loads the data
train_dataset, test_dataset = datasets['train'], datasets['test'] # Splits the dataset 
# print(metadata)

class_name = metadata.features['label'].names # Prints the names of the labels
print(class_name)
print(metadata.features) # Prints the features


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# for image, label in test_dataset.take(1): # this example is to check out one of the examples
#   break
# image = image.numpy().reshape((28,28))

# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show() # to check out one of the example shirts

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #flattens the 3d image into one array of numbers
    tf.keras.layers.Dense(512, activation=tf.nn.relu), #creates 128 level of neural networks
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #softmax produces probability distribution 
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) 

BATCH_SIZE = 32 #Batch size is the number of iterations the neural network does before it is updated 
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE) #Updates dataset with batch included
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_test_examples/32)) # This trains the model 

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
print('Loss: ', test_loss)

for test_images, test_labels in test_dataset.take(1): #predict one sample from the data
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

predictions.shape
print(np.argmax(predictions[0]))
# print(test_labels[0]) # If this line and the line above returns the same value, then it is a correct answer. Otherwise the answer is wrong.
# print(class_name[test_labels[0]])
