"""
@author: P_k_y
"""

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import load_model
import tkinter.messagebox as msg
import numpy as np


class NN:

    def __init__(self):
        self.model = None

    def create_model(self, input_shape, class_number):
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=input_shape))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(class_number, activation='softmax'))
        self.model.compile(optimizer=SGD(), loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        if self.model is None:
            msg.showwarning("Warning", "Please create a model before fit it.")
            return
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    def save_model(self, name):
        self.model.save(name)
        print("\n->Model has been saved as: " + name)

    def load_model(self, name):
        self.model = load_model(name)

    def predict(self, X_test):
        result_list = []
        result_class_number = 0

        print("\nPredicting....\n")
        for i in range(len(X_test)):
            x = X_test[i].reshape(-1, len(X_test[i]))
            result = self.model.predict(x)[0]
            result_list.append(np.argmax(result))
            if result_class_number == 0:
                result_class_number = len(result)

        print("\n-> The Prediction Result is: ")
        for result in result_list:
            print(result, end='')
        print("\n   <", end='')
        for i in range(result_class_number):
            print("  %d : %3d  |" % (i, result_list.count(i)), end='')
        print("   >\n")

    def predict_image(self, X_test, image_size):
        result_list = []
        print("\nPredicting....\n")
        for i in range(len(X_test)):
            x = X_test[i].reshape(-1, len(X_test[i]))
            result = self.model.predict(x)
            result_list.append(np.argmax(result))
        result_list = np.array(result_list)
        return result_list.reshape(image_size)

