import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50


def build_latent_layer(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, y_train, x_test, y_test


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

mean = Dense(latent_dim)(h)
log_var = Dense(latent_dim)(h)

z = Lambda(build_latent_layer, output_shape=(latent_dim,))([mean, log_var])

# Build the hidden layer decoder, mean decoder which will then
# transform the reduced space into original dimension
hidden_layer_decoder = Dense(intermediate_dim, activation='relu')
# use sigmoid because we want the probability of activation of neurons
mean_decoder = Dense(original_dim, activation='sigmoid')
decoded_hidden_layer = hidden_layer_decoder(z)
decoded_mean = mean_decoder(decoded_hidden_layer)

vae = Model(x, decoded_mean)

# calculate cross entropy and reconstruction loss
binary_cross_entropy = K.sum(K.binary_crossentropy(x, decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
vae_loss = K.mean(binary_cross_entropy + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
x_train, _, x_test, _ = get_mnist_data()
vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

encoder = Model(x, mean)


def main():
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = hidden_layer_decoder(decoder_input)
    _x_decoded_mean = mean_decoder(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    main()
