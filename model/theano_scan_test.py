import theano as theano
import theano.tensor as T
import numpy as np
from sklearn.cluster import KMeans

x_train = np.array([[1., 1.],
                    [1., 0.],
                    [1., 1.],
                    [0., 0.]])

initial_centers = np.array([[0., 0.],
                            [1., 1.],
                            [1., 0.]])
print x_train.shape
print initial_centers.shape
# a_tile = T.tile(a, (b.shape[0], 1))
# b_tile = T.tile(b, (1, a.shape[1]))
# print a_tile
# print b_tile

x_batch = T.dmatrix("x_batch")
centers_all = T.dmatrix("centers")


# cc = aa + bb
# cc = th.dot(aa, bb)
# cc = T.tile(aa, (bb.shape[0], 1)) * T.tile(bb, (1, aa.shape[1]))


def P1(z, centers):
    dst, _ = theano.scan(lambda z_i: T.sqrt(((z_i - centers) ** 2).sum(axis=1)),
                         sequences=z)
    m_dst = dst.mean(axis=1)
    q = T.maximum(0.0, T.tile(m_dst, (dst.shape[1], 1)).T - dst)

    num_centers = q.shape[1]
    weight = 1.0 / q.sum(axis=0)
    weight *= num_centers / weight.sum()
    q = (q ** 2.0) * weight

    q = (q.T / q.sum(axis=1)).T
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
    cosine_simi = dot_product / (norm_product + 1e-9)
    q = 1. / (2. - cosine_simi)
    num_centers = q.shape[1]
    weight = 1.0 / q.sum(axis=0)
    weight *= num_centers / weight.sum()
    q = (q ** 2.0) * weight
    q = (q.T / q.sum(axis=1)).T
    return q


p1 = P1(x_batch, centers_all)
p2 = P2(x_batch, centers_all)
loss = T.sum((p2 + 1e-9) * T.log((p2 + 1e-9) / (p1 + 1e-9)), axis=1)

# ff = theano.function([x_batch, centers_all], p1, on_unused_input='warn')
f1 = theano.function([x_batch, centers_all], p1, on_unused_input='warn')
f2 = theano.function([x_batch, centers_all], p2, on_unused_input='warn')

print "f1"
print f1(x_train, initial_centers)

print "f2"
print f2(x_train, initial_centers)

loss_function = theano.function([x_batch, centers_all], loss, on_unused_input='warn')
print "loss"
print loss_function(x_train, initial_centers)
