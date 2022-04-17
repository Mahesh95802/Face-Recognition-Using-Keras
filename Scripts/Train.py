from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from Scripts.CreateLabels import createLabels

def train(db):
    vgg = VGG16(input_shape=[200, 200] + [3], weights='imagenet', include_top=False)
    label_ids = createLabels(db)
    for layer in vgg.layers:
        layer.trainable = False   
    x = Flatten()(vgg.output)
    prediction = Dense(len(os.listdir(os.path.join(db, "Train"))), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    print(model.summary())
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory(os.path.join(db, "Train"),
                                                    target_size = (200, 200),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(os.path.join(db, "Test"),
                                                target_size = (200, 200),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    r = model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=12,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set))
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')
    print(r.history)
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')
    model.save('FaceRecognitionModel.h5')
    print("Model has been Saved.")
    f = open("MyFile2.txt","w")
    f.write(str(r.history))
    f.close()
