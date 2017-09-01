from keras.datasets import mnist
import numpy as np
from scipy import spatial

# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train = np.array([[1, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1]])

initial_centers = np.array([[0, 0, 0],
                            [1, 1, 1]])
# let
# n train samples
# k cluster centers
# m feature dim

# n * m
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
non_center = x_train
# k * m
initial_centers = initial_centers.reshape((len(initial_centers), np.prod(initial_centers.shape[1:])))

# n * k
z = np.array(np.sum(np.power(np.subtract(non_center, initial_centers[0]), 2), 1))
for index in range(1, initial_centers.shape[0]):
    col = np.sum(np.power(np.subtract(non_center, initial_centers[index]), 2), 1)
    z = np.vstack((z, col))
z = z.transpose()
# print z
# z = np.subtract(non_center, initial_centers)
# print z

# n * 1
u = np.mean(z, 1).reshape(len(non_center), 1)
print "z:"
print z
print "u:"
print u
#
sub = np.subtract(u, z)
print "sub:"
print sub
#
Q1 = np.maximum(0, sub)
print "Q1:"
print Q1

#
# test1 = np.array([[1], [0], [1]]).reshape(3, 1)
# test2 = np.array([[2], [1]]).reshape(1, 2)
# test_mul = np.multiply(test1, test2)
# print "test_mul:"
# print test_mul

dot_product = np.dot(non_center, np.transpose(initial_centers))
print "dot product:"
print dot_product

non_center_norm = np.reshape(np.power(np.sum(np.power(non_center, 2), 1), 0.5), (non_center.shape[0], 1))
initial_centers_norm = np.reshape(np.power(np.sum(np.power(initial_centers, 2), 1), 0.5), (1, initial_centers.shape[0]))
norm_product = np.multiply(non_center_norm, initial_centers_norm)

print "non_center_norm"
print non_center_norm
print "initial_cneter_norm"
print initial_centers_norm

print "norm product"
print norm_product

cosine_simi = np.divide(dot_product, norm_product + 1e-9)
Q2 = np.divide(1, np.subtract(2, cosine_simi))

print "Q2"
print Q2