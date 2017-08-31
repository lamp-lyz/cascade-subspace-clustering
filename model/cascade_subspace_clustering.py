from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import kullback_leibler_divergence
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded_layer_2 = Dense(500, activation='relu')(input_img)
encoded_layer_3 = Dense(500, activation='relu')(encoded_layer_2)
encoded_layer_4 = Dense(2000, activation='relu')(encoded_layer_3)
encoded = Dense(encoding_dim, activation='relu')(encoded_layer_4)

# "decoded" is the lossy reconstruction of the input
decoded_layer_1 = Dense(encoding_dim, activation='relu')(encoded)
decoded_layer_2 = Dense(2000, activation='relu')(decoded_layer_1)
decoded_layer_3 = Dense(500, activation='relu')(decoded_layer_2)
decoded_layer_4 = Dense(500, activation='relu')(decoded_layer_3)
decoded = Dense(784, activation='sigmoid')(decoded_layer_4)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-4]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

k_means = KMeans(n_clusters=10)
initial_centors = k_means.fit(encoded).cluster_centers_
clustering_model = Model(input_img, encoded)

u = 
y_true = max(0,)
y_pred = 1

lossfunc = kullback_leibler_divergence(y_true, y_pred)

clustering_model
