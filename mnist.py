import tensorflow as tf
import numpy as np
import seaborn as sb 
from matplotlib import pyplot as plt
import sys
import os

EPOCHS = 5

cpPath = os.path.dirname(os.path.realpath(__file__)) + "\mnistcallback"

def main():

    # build model
    trainX, testX, trainY, testY = load_dataset()
    model = build_model(trainX, testX, trainY, testY)

    # load the model and use it to predict values
    # model = tf.keras.models.load_model(cpPath)
    # predict(testX, testY, model)
    
    model.summary()

def load_dataset():

    # load dataset
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    # print(trainX.shape)

    # reshape the data
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    # range of values is 0 - 1
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    return trainX, testX, trainY, testY


def build_model(trainX, testX, trainY, testY):
    
    # 4 layers
    # Layer 1 : Input layer (input: 784, output: 512)
    # Layer 2 : Hidden layer 1 (input: 512, output: 256)
    # Layer 3 : Hidden layer 2 (input: 256, output: 128)
    # Layer 4 : Output layer (input: 128, output: 10)

    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Dense(512, input_shape=(784,)),
    #         tf.keras.layers.Dense(256, activation = "relu"),
    #         tf.keras.layers.Dense(128, activation = "relu"),
    #         tf.keras.layers.Dense(10, activation="sigmoid"),
    #     ]
    # )

    # Convolutional Neural Network
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    
    # compile
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # save the weights 
    callback = tf.keras.callbacks.ModelCheckpoint(filepath = cpPath,
                                                 save_weights_only = True,
                                                 verbose=1)

    model.fit(trainX, trainY, epochs=EPOCHS)
    model.evaluate(testX, testY)
    
    # save the model
    model.save(cpPath)

    return model


def predict(testX, testY, model):

    predictY = model.predict(testX)

    yPredictLabels = [np.argmax(i) for i in predictY]

    # visualise the results
    cm = tf.math.confusion_matrix(labels=testY, predictions=yPredictLabels)
    print(cm)

    plt.figure(figsize = (10,7))
    sb.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

main()
