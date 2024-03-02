import tensorflow as tf
import os

checkpoint_path = r"C:\Programming\Deep-Learning-Fashion-Classifier\Model\training_2\cp-0010.ckpt.data-00000-of-00001"
checkpoint_dir = os.path.dirname(checkpoint_path)

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


