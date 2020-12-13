import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class GAN:
    def __init__(self):
        self.rows = 28
        self.cols = 28
        self.img_shape = (self.rows, self.cols, 1)

        optimizer = Adam(0.0002, 0.5)
        self.disc = self.get_disc_nn()
        self.disc.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
        self.generator = self.get_gen_nn()
        self.input_noise_dim = 100

        gan_input = Input(shape=(self.input_noise_dim,))
        img = self.generator(gan_input)
        self.disc.trainable = False
        validity = self.disc(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def add_hidden_layer(self, model, num_nodes, normalize=False, is_input=False):
        if is_input:
            model.add(Dense(num_nodes, input_dim=self.input_noise_dim))
        else:
            model.add(Dense(num_nodes))
        model.add(LeakyReLU(alpha=0.2))

        if normalize:
            model.add(BatchNormalization(momentum=0.8))

    def get_gen_nn(self):
        model = Sequential()

        self.add_hidden_layer(model, 256, True, True)
        self.add_hidden_layer(model, 512, True)
        self.add_hidden_layer(model, 1024, True)

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.input_noise_dim,))
        img = model(noise)

        return Model(noise, img)

    def get_disc_nn(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))

        self.add_hidden_layer(model, 512)
        self.add_hidden_layer(model, 256)

        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.input_noise_dim))

            output = self.generator.predict(noise)
            d_loss_real = self.disc.train_on_batch(imgs, valid)
            d_loss_fake = self.disc.train_on_batch(output, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.input_noise_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.generate_images(epoch)

    def generate_images(self, epoch):
        row, column = 5, 5
        noise = np.random.normal(0, 1, (row * column, self.input_noise_dim))
        output = 0.5 * self.generator.predict(noise) + 0.5

        fig, axs = plt.subplots(row, column)
        cnt = 0
        for i in range(row):
            for j in range(column):
                axs[i, j].imshow(output[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("gan/%d.png" % epoch)
        plt.close()


def main():
    if not os.path.exists("./gan"):
        os.makedirs("./gan")
    gan = GAN()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)


if __name__ == '__main__':
    main()

