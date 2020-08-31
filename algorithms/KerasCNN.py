from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt


class KerasCNN:
    def __init__(self, input_shape, iterations):
        self.model = self.get_model(input_shape)
        self.history = None
        self.iterations = iterations
        self.model.summary()

    def fit(self, X_train, Y_train, optimizer='adam'):
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x=X_train.T, y=Y_train.T, epochs=self.iterations, batch_size=256)

        self.model.save_weights('./outputs/tf-cnn-params.h5')

    def get_accuracy(self, X_test, Y_test, type=''):
        preds = self.model.evaluate(x=X_test, y=Y_test)
        accuracy = str(preds[1])

        print('    -', type, "Accuracy:", accuracy, '%')

    def load_params(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.load_weights('./outputs/tf-cnn-params.h5')

    def plot_cost(self):
        cost = self.history.history['loss']

        plt.plot(cost)
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.title("Learning Curve")
        plt.show()

    def get_model(self, input_shape):
        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = Activation('relu')(X)

        X = MaxPooling2D((2, 2), name='max_pool')(X)

        X = Flatten()(X)
        X = Dense(10, activation='softmax', name='fc')(X)

        model = Model(inputs=X_input, outputs=X, name='TF-CNN')

        return model
