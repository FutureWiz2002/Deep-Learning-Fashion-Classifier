import tensorflow as tf
import os
import tensorflow_datasets as tfds
import math

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

datasets, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True) # Loads the data
train_dataset, test_dataset = datasets['train'], datasets['test'] # Splits the dataset

BATCH_SIZE = 32
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

checkpoint_path = "training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
num_test_examples = metadata.splits['test'].num_examples

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

model.load_weights(checkpoint_path)

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print("Restored model, accuracy: {:5.2f}%".format(100 * test_accuracy))
