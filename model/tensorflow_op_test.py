from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import kullback_leibler_divergence
from keras import losses
from keras import backend as K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf

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

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

batch_size = 256

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

x_train_embedding = encoder.predict(x_train)
k_cluster = 10
k_means = KMeans(n_clusters=k_cluster)
initial_centers = k_means.fit(x_train_embedding).cluster_centers_

print initial_centers

# non_center = Input(shape=(None, encoding_dim))


# batch_size * k_cluster each line is dist(zi, u)
z = np.array(np.sum(np.power(np.subtract(x_train_embedding, initial_centers[0]), 2), 1))
for index in range(1, initial_centers.shape[0]):
    col = np.sum(np.power(np.subtract(x_train_embedding, initial_centers[index]), 2), 1)
    z = np.vstack((z, col))
z = z.transpose()
print z
# batch_size * 1 each line is mean(dist(zi,u))
u = np.mean(z, 1).reshape(len(x_train_embedding), 1)

# Q1 Euclidean dissimilarity measure
Q1 = np.maximum(0, np.subtract(u, z))

dot_product = np.dot(x_train_embedding, np.transpose(initial_centers))

# Q2 Cosine dissimilarity measure
x_train_embedding_norm = np.reshape(np.power(np.sum(np.power(x_train_embedding, 2), 1), 0.5), (x_train_embedding.shape[0], 1))
initial_centers_norm = np.reshape(np.power(np.sum(np.power(initial_centers, 2), 1), 0.5), (1, initial_centers.shape[0]))
norm_product = np.multiply(x_train_embedding_norm, initial_centers_norm)

cosine_simi = np.divide(dot_product, norm_product + 1e-9)
Q2 = np.divide(1, np.subtract(2, cosine_simi))
#
# P1 =
#
# P2 =
loss = losses.binary_crossentropy(Q1, Q2)


# lossfunc = kullback_leibler_divergence(P2, P1)
# output = Dense(10, activation="sigmoid", )
cluster_input = Input(shape=(encoding_dim,))
clustered = Dense(encoding_dim, activation=None)(encoded)
clustering_model = Model(cluster_input, clustered)
clustering_model.compile(optimizer=SGD(lr=0.01, momentum=0.9,decay=1e-6), loss=loss)
clustering_model.fit(x_train, x_train,
                     epochs=5,
                     batch_size=batch_size,
                     shuffle=True)
