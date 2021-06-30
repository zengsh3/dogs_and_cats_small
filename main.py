from keras.applications import VGG16
from keras import optimizers
from keras import models
from keras import layers
from keras.models import load_model
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

train_dir = 'dataset\\train'
validation_dir = 'dataset\\validation'
test_dir = 'dataset\\test'

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
model.save('cats_and_dogs_small_20210701.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.figure(figsize=(10,12))

plt.subplot(211)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.subplot(212)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(fname='original picture.png')


def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.figure(figsize=(10,12))
plt.subplot(211)
plt.plot(epochs,smooth_curve(acc), 'bo', label='training acc')
plt.plot(epochs,smooth_curve(val_acc), 'b', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(212)
plt.plot(epochs,smooth_curve(loss), 'bo', label='training loss')
plt.plot(epochs,smooth_curve(val_loss), 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(fname='smoothed picture.png')


test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),
                                                  batch_size=20,class_mode='binary')

model=load_model('cats_and_dogs_small_20210701.h5')
test_loss,test_acc = model.evaluate_generator(test_generator,steps=50)
print('test_acc:',test_acc)
