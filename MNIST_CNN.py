import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)
print(train_images.shape)
print(test_images.shape)

def build_cnn_model():
    cnn_model = tf.keras.Sequential([                                    #构建Model的第二种方式，Sequential（，，，，）顺序结构，也可model = modles.sequential() model.add...
        tf.keras.layers.Conv2D(filters=24, kernel_size=(5,5), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(filters=36, kernel_size=(5,5), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return cnn_model

cnn_model = build_cnn_model()

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the batch size and the number of epochs to use for training
BATCH_SIZE = 64
EPOCHS = 5

#train_image = train_images.reshape(len(train_images),-1)
#train_label = to_categorical(train_labels)

history = cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, validation_split=0.2, shuffle=True)
print(history.history.keys())

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(fname='history')

test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

preds = cnn_model.predict(test_images, BATCH_SIZE)


def plot_image_prediction(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(np.squeeze(img), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
               color=color)


def plot_value_prediction(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


image_index = 0  # @param {type:"slider", min:0, max:100, step:1}

plt.figure()
plt.subplot(1, 2, 1)
plot_image_prediction(image_index, preds, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_prediction(image_index, preds, test_labels)
plt.savefig(fname='prediction')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import pandas as pd
import seaborn as sn

# convert from one-hot vector to precticted labels
preds_labels = np.argmax(preds, axis=1)
confusion_matrix = confusion_matrix(test_labels, preds_labels)
# print(confusion_matrix)

# plot confusion matrix
plt.figure()
df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 10})
sn.set(font_scale=0.8)


plt.ylabel('labels')
plt.xlabel('predictions')
plt.savefig(fname='confusion_matrix')
