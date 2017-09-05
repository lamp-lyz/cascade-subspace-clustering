from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.datasets import mnist
from keras import backend as T
from sklearn.cluster import KMeans
import numpy as np
from keras.optimizers import SGD
import theano as theano
import hungarian as hg

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

batch_size = 256

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


x_train_embedding = Input(shape=(encoding_dim,))
k_cluster = 10
k_means = KMeans(n_clusters=k_cluster)
encoded_predict = encoder.predict(x_train)
initial_centers = k_means.fit(encoded_predict).cluster_centers_

encoded_test = encoder.predict(x_test)
kmeans_label = k_means.predict(encoded_test)
true_label = y_test
rectified_label, kmeans_acc = hg.rectify_label(labels=kmeans_label, classes=true_label)
print "rectified label"
print rectified_label

print "kmeans accuracy"
print kmeans_acc


# print "initial centers:"
# print initial_centers.shape


def myloss(y_true, y_pred):
    # encoded_predict = encoder.predict(x_train)
    # centers = k_means.fit(encoded_predict).cluster_centers_
    # print "center shape"
    # print centers.shape
    # print "y pred shape"
    # print y_pred.shape
    p1 = P1(y_pred, initial_centers)
    print p1

    p2 = P2(y_pred, initial_centers)
    print p2

    loss = T.sum(((p2 + 1e-9) * T.log((p2 + T.epsilon()) / (p1 + T.epsilon()))), axis=1)
    return loss

    # print "y pred origin"
    # print y_pred
    # y_pred_reshape = tf.reshape(y_pred, (-1, 1, encoding_dim))
    # print "y pred"
    # print y_pred
    # temp_sub = tf.subtract(y_pred_reshape, centers)
    # print "temp sub"
    # print temp_sub
    # temp_pow = tf.pow(temp_sub, 2)
    # print "temp pow"
    # print temp_sub
    # z = tf.reshape(tf.pow(tf.reduce_sum(temp_pow, 2), 0.5), (-1, 1, k_cluster))
    # print "z:"
    # print z
    # # n * 1
    # u = tf.reduce_mean(z, 1)
    # print "u:"
    # print u
    #
    # sub = tf.subtract(u, z)
    # print "sub:"
    # print sub
    #
    # Q1 = tf.maximum(tf.cast(1e-9, tf.float32), sub)
    # print "Q1:"
    # print Q1
    #
    # dot_product = tf.matmul(y_pred, tf.transpose(centers))
    # print "dot product:"
    # print dot_product
    #
    # non_center_norm = tf.reshape(tf.pow(tf.reduce_sum(tf.pow(y_pred, 2), 1), 0.5), (-1, 1))
    # initial_centers_norm = tf.reshape(tf.pow(tf.reduce_sum(tf.pow(centers, 2), 1), 0.5), (1, -1))
    # norm_product = tf.multiply(non_center_norm, initial_centers_norm)
    #
    # print "non_center_norm"
    # print non_center_norm
    # print "initial_cneter_norm"
    # print initial_centers_norm
    #
    # print "norm product"
    # print norm_product
    #
    # cosine_simi = tf.div(dot_product, tf.add(norm_product, tf.cast(1e-9, tf.float32)))
    # print "cosine simi:"
    # print cosine_simi
    #
    # Q2 = tf.div(tf.cast(1, tf.float32), tf.subtract(tf.cast(2, tf.float32), cosine_simi))
    #
    # print "Q2"
    # print Q2
    # los = K.sum(Q2 * K.log(Q2 / Q1), axis=-1)
    # return los


def P1(z, centers):
    dst, _ = theano.scan(lambda z_i: T.sqrt(((z_i - centers) ** 2).sum(axis=1)),
                         sequences=z)
    m_dst = dst.mean(axis=1)
    q = T.maximum(0.0, T.tile(m_dst, (dst.shape[1], 1)).T - dst)

    num_centers = q.shape[1]
    weight = 1.0 / (q.sum(axis=0) + T.epsilon())
    weight *= num_centers / weight.sum()
    q = (q ** 2.0) * weight

    q = (q.T / (q.sum(axis=1) + T.epsilon())).T
    return q


def P2(z, centers):
    # dot_product = tf.matmul(z, tf.transpose(centers))
    # non_center_norm = tf.reshape(tf.pow(tf.reduce_sum(tf.pow(z, 2), 1), 0.5), (-1, 1))
    # initial_centers_norm = tf.reshape(tf.pow(tf.reduce_sum(tf.pow(centers, 2), 1), 0.5), (1, -1))
    # norm_product = tf.multiply(non_center_norm, initial_centers_norm)
    # cosine_simi = tf.div(dot_product, tf.add(norm_product, tf.cast(1e-9, tf.float32)))
    # q = tf.div(tf.cast(1, tf.float32), tf.subtract(tf.cast(2, tf.float32), cosine_simi))

    dot_product = theano.dot(z, centers.T)
    non_center_norm = T.sqrt((z ** 2).sum(axis=1))
    initial_centers_norm = T.sqrt((centers ** 2).sum(axis=1))
    non_center_norm_tile = T.tile(non_center_norm, (centers.shape[0], 1)).T
    center_norm_tile = T.tile(initial_centers_norm, (z.shape[0], 1))
    norm_product = non_center_norm_tile * center_norm_tile
    cosine_simi = dot_product / (norm_product + T.epsilon())
    q = 1. / (2. - cosine_simi)
    num_centers = q.shape[1]
    weight = 1.0 / (q.sum(axis=0) + T.epsilon())
    weight *= num_centers / weight.sum()
    q = (q ** 2.0) * weight
    q = (q.T / (q.sum(axis=1) + T.epsilon())).T
    return q


# print "myloss"
# print myloss
# non_center = Input(shape=(None, encoding_dim))

#
# # lossfunc = kullback_leibler_divergence(P2, P1)
# output = Dense(10, activation="sigmoid", )
clustering_model = Model(input_img, encoded)
clustering_model.compile(optimizer=SGD(lr=1e-5, momentum=0.9, decay=1e-6), loss=myloss)
# clustering_model.compile(optimizer='adadelta', loss=myloss)

clustering_model.fit(x=x_train,
                     y=x_train,
                     epochs=500,
                     batch_size=batch_size,
                     shuffle=True)


def assginment(clustering_results, cluster_centers):
    print "result"
    print clustering_result
    print "center"
    print cluster_centers

    dst = np.array(np.sum(np.power(np.subtract(clustering_results, cluster_centers[0]), 2), 1))
    for index in range(1, initial_centers.shape[0]):
        col = np.sum(np.power(np.subtract(clustering_results, cluster_centers[index]), 2), 1)
        print "temp dst"
        print col.shape

        dst = np.vstack((dst, col))
    dst = dst.T
    print "dst"
    print dst.shape
    mean_dst = np.mean(dst, axis=1)
    q = np.maximum(0.0, np.tile(mean_dst, (dst.shape[1], 1)).T - dst)

    num_centers = q.shape[1]
    weight = 1.0 / (q.sum(axis=0) + 1e-7)
    weight *= num_centers / weight.sum()
    q = (q ** 2.0) * weight

    q = (q.T / (q.sum(axis=1) + T.epsilon())).T
    return q


clustering_result = clustering_model.predict(x_test)
print "clustering result"
print clustering_result

assignment = assginment(clustering_result, initial_centers)
print "assignment:"
print assignment.shape

css_labels_pred = np.argmax(assignment, axis=1)
print "css label pred"
print css_labels_pred.shape
classes = y_test

(css_labels, css_accuracy) = hg.rectify_label(labels=css_labels_pred, classes=classes)
print "cascade subspace clustering accuracy"
print css_accuracy
#
# km_labels_pred = k_means.predict(encoded_test)
#
# (km_labels, km_accuracy) = hg.rectify_label(labels=km_labels, classes=classes)
#
# print "kmeans clustering accuracy"
# print km_accuracy