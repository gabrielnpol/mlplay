
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

df = pd.read_csv(r"./data/trainLabels.csv")
train_generator = datagen.flow_from_dataframe(dataframe=df, directory="./data/train/train", x_col="id",
                                              y_col="label", has_ext=False, class_mode="categorical",
                                              target_size=(32, 32), batch_size=32)
X_input = Input((32, 32, 3))
X = ZeroPadding2D((3, 3))(X_input)
X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
X = BatchNormalization(axis=3, name='bn0')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool')(X)
X = Flatten()(X)
X = Dense(10, activation='softmax', name='fc')(X)

model = Model(inputs=X_input, outputs=X, name='Recogn')
steps = train_generator.n//train_generator.batch_size

print(steps)

model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=["accuracy"])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps,
                    epochs=1)
