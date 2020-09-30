from keras.layers import (Convolution2D, Input, Dense, MaxPooling2D,
                          Dropout, Activation, BatchNormalization, Flatten)
from keras.models import Model, load_model
from keras.utils import plot_model

# Additional Scripts
import config as CFG


class MTLAgeGender:
    input_shape = (CFG.INPUT_SHAPE, CFG.INPUT_SHAPE, 1)

    def build_model(self, inp):
        conv1 = Convolution2D(32, (3, 3), padding="same")(inp)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D((2, 2), padding="same")(conv1)

        conv2 = Convolution2D(64, (3, 3), padding="same")(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D((2, 2), padding="same")(conv2)

        conv3 = Convolution2D(128, (3, 3), padding="same")(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D((2, 2), padding="same")(conv3)

        conv4 = Convolution2D(256, (3, 3), padding="same")(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D((2, 2), padding="same")(conv4)

        flatten = Flatten()(pool4)

        return flatten

    def model(self):
        x_input = Input(self.input_shape)

        flatten = self.build_model(x_input)

        output_gender = Dense(256, activation="relu")(flatten)
        output_gender = Dropout(0.3)(output_gender)
        output_gender = Dense(128, activation="relu")(output_gender)
        output_gender = Dropout(0.3)(output_gender)
        output_gender = Dense(1, activation="sigmoid", name='gender')(output_gender)

        output_age = Dense(256, activation='relu')(flatten)
        output_age = Dropout(0.3)(output_age)
        output_age = Dense(128, activation='relu')(output_age)
        output_age = Dropout(0.3)(output_age)
        output_age = Dense(5, activation='softmax', name='age')(output_age)

        model = Model(inputs=[x_input], outputs=[output_gender, output_age],
                      name="mtl_age_gender")

        return model


class MTLAgeGenderEmotion:
    input_shape = (64, 64, 1)

    def get_age_gender_model(self):
        model = load_model(CFG.PRETRAINED_AGE_GENDER_MODEL_PATH)
        for layer in model.layers:
            layer.trainable = True

        flatten = model.get_layer('flatten_1').output
        output_age = model.get_layer('age').output
        output_gender = model.get_layer('gender').output

        return model, flatten, output_age, output_gender

    def model(self):
        model, flatten, output_age, output_gender = self.get_age_gender_model()

        output_emotion = Dense(256, activation='relu', name='emo_dense_1')(flatten)
        output_emotion = Dropout(0.3, name='emo_drop_1')(output_emotion)
        output_emotion = Dense(128, activation='relu', name='emo_dense_2')(output_emotion)
        output_emotion = Dropout(0.3, name='emo_drop_2')(output_emotion)
        output_emotion = Dense(7, activation='softmax', name='emotion')(output_emotion)

        model = Model(inputs=[model.input], outputs=[output_gender, output_age, output_emotion],
                      name="mtl_age_gender_emotion")

        return model


if __name__ == '__main__':
    ann = MTLAgeGender().model()
    ann.summary()
    plot_model(ann, to_file="age_gender_mtl_model.png")

    ann = MTLAgeGenderEmotion().model()
    ann.summary()
    plot_model(ann, to_file='age_gender_emotion_mtl_model.png')
