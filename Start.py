"""
@author: P_k_y
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import os
import pandas as pd
from NN import NN as NeuralNetework
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from ColorMapping import ColorMapping
import cv2
import tkinter.messagebox as msg

WIDTH = 720
HEIGHT = 500

train_dataset = None
train_dataset_len = None

train_dim = None
class_number = None

X = None
y = None

NN = None


def get_train_file():
    global train_dataset, train_dataset_len, X, y, train_dim, class_number

    train_file_path = filedialog.askopenfilename()
    if train_file_path is not "":
        df = pd.read_csv(train_file_path, header=None)
        train_dataset = df.values
        train_dataset_len = train_dataset.shape[0]
        X = train_dataset[:, 1:]
        y = train_dataset[:, 0]
        train_dim = (X.shape[1], )
        class_number = max(y) + 1
        file_dir, file_name = os.path.split(train_file_path)
        train_file_label["text"] = "Train Data File:        " + file_name + " (Len: %d)" % train_dataset_len


def start():
    global train_dataset_len, train_dim, class_number, X, y, NN
    test_size = simpledialog.askfloat("Set test size", "Input the test size(0 ~ 0.9), which means the size of samples used to test model's accuracy: ", initialvalue=0.2, minvalue=0, maxvalue=0.9)
    test_samples = int(train_dataset_len) * test_size
    train_samples = train_dataset_len - test_samples
    info_label["text"] = "Train Data Size:        %6d\nTest Data Size:         %6d" % (train_samples, test_samples)
    epoch = simpledialog.askinteger("Set Epoch", "Input the Epochs(0~INF), which mean the iteration of training: ", initialvalue=8, minvalue=1)

    max_value = np.max(X)
    print(max_value)
    with open("max_value.txt", 'w') as f:
        f.write(str(max_value))

    X = X / max_value
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    NN = NeuralNetework()
    NN.create_model(train_dim, class_number)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    NN.train(X_train, y_train, X_test, y_test, epochs=epoch, batch_size=1)
    print("\nFinished Model Training! Press 'Save Model' Button to save this model.\n")


def save_model():
    global NN
    model_name = simpledialog.askstring("Set model name", "Input the name of saving model: ", initialvalue="model1")
    saved_name = model_name + ".h5"
    NN.save_model(saved_name)


def main():
    win.mainloop()


def get_model_file():
    global NN
    model_file_name = filedialog.askopenfilename()
    if model_file_name is not "":
        NN = NeuralNetework()
        NN.load_model(model_file_name)
        model_file_label["text"] = "Model File:        " + os.path.split(model_file_name)[1]


def get_test_file():
    global NN

    if NN is not None and NN.model is not None:
        pass
    else:
        msg.showwarning("Warning", "Need to load model before Predict!")
        return

    test_file_name = filedialog.askopenfilename()
    if test_file_name is not "":
        df = pd.read_csv(test_file_name, header=None)
        test_dataset = df.values
        test_dataset_len = test_dataset.shape[0]
        test_file_label["text"] = "Test File:        " + os.path.split(test_file_name)[1] + " (Len: %d)" % test_dataset_len
        with open("max_value.txt", 'r') as f:
            max_value = int(f.read())
            print("Max Value in train dataset is:" + str(max_value))
        test_dataset = test_dataset / max_value
        NN.predict(test_dataset)


def get_image_file():
    global NN

    if NN is not None and NN.model is not None:
        pass
    else:
        msg.showwarning("Warning", "Need to load model before Predict!")
        return

    test_img_name = filedialog.askopenfilename()
    if test_img_name is not "":
        df = pd.read_csv(test_img_name, header=None)
        test_dataset = df.values
        test_dataset_len = test_dataset.shape[0]
        image_file_label["text"] = "Predict Image File:        " + os.path.split(test_img_name)[1] + " (Len: %d)" % test_dataset_len
        with open("max_value.txt", 'r') as f:
            max_value = int(f.read())
            print("Max Value in train dataset is:" + str(max_value))
        test_dataset = test_dataset / max_value

        str_img_shape = simpledialog.askstring("Set Image Size", "Input Image Size(ex: 320x240): ", initialvalue="320x240")
        img_shape = (int(str_img_shape.split('x')[1]), int(str_img_shape.split('x')[0]))
        mat_matrix = NN.predict_image(test_dataset, img_shape)

        cm = ColorMapping(mat_matrix)
        img = cm.map_color()
        cv2.imshow("Result Image", img)
        cv2.waitKey(0)
        cm.save_color_image("result.jpg")
        print("\n->Result Image saved as: result.jpg")


win = tk.Tk()
win.title("Neural Network Workbench For Spectrum")
win.geometry("{}x{}".format(WIDTH, HEIGHT))
title = tk.Label(win, text="Neural Network Workbench For Spectrum v1.0", bg="green", fg="white", font=('Arial', 14), height=3, width=65)
title.grid(row=1, column=1, columnspan=3)
train_label = tk.Label(win, text="- Train Area -", font=('Arial', 12), height=3)
train_label.grid(row=2, column=1, columnspan=3)
train_file_label = tk.Label(win, text="Train Data File:        (Not Select)", font=('Arial', 11), height=3)
train_file_label.grid(row=3, column=1, columnspan=2, sticky='w', ipadx=50)
get_train_file_button = tk.Button(win, text="Choose Train File", command=get_train_file, width=20)
get_train_file_button.grid(row=3, column=3)
info_label = tk.Label(win, text="Train Data Size:        (Not Select)\nTest Data Size:         (Not Select)", font=('Arial', 11), height=3)
info_label.grid(row=4, column=1, columnspan=2, rowspan=2, sticky='w', ipadx=50)
start_button = tk.Button(win, text="Start Train", command=start, width=20)
start_button.grid(row=4, column=3)
save_model_button = tk.Button(win, text="Save Model", command=save_model, width=20)
save_model_button.grid(row=5, column=3)

test_label = tk.Label(win, text="- Test Area -", font=('Arial', 12), height=3)
test_label.grid(row=6, column=1, columnspan=3)
model_file_label = tk.Label(win, text="Model File:        (Not Select)", font=('Arial', 11), height=3)
model_file_label.grid(row=7, column=1, columnspan=2, sticky='w', ipadx=50)
get_model_button = tk.Button(win, text="Choose Model File", command=get_model_file, width=20)
get_model_button.grid(row=7, column=3)

test_file_label = tk.Label(win, text="Test File:        (Not Select)", font=('Arial', 11), height=3)
test_file_label.grid(row=8, column=1, columnspan=2, sticky='w', ipadx=50)
get_test_button = tk.Button(win, text="Choose Test File & Start", command=get_test_file, width=20)
get_test_button.grid(row=8, column=3)

image_file_label = tk.Label(win, text="Predict Image File:        (Not Select)", font=('Arial', 11), height=3)
image_file_label.grid(row=9, column=1, columnspan=2, sticky='w', ipadx=50)
get_image_button = tk.Button(win, text="Choose Image & Start", command=get_image_file, width=20)
get_image_button.grid(row=9, column=3)

if __name__ == '__main__':
    main()
