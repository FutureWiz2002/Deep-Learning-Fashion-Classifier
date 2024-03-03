import tensorflow as tf
import os
import tensorflow_datasets as tfds

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

checkpoint_path = r"training_2\cp-0010.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)


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

model.load_weights(latest).expect_partial()

loss, acc = model.evaluate(train_images, test_images, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


