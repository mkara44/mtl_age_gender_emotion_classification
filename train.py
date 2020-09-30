import argparse
import pandas as pd
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Additional Scripts
from utils.utils import create_folders
from utils.generator import DataGenerator
from utils.nn import MTLAgeGender, MTLAgeGenderEmotion
from utils.callbacks import DrawGraph, MultiOutputEarlyStoppingAndCheckpoint

import config as CFG

import tensorflow as tf
from keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
set_session(sess)


class Training:
    batch_size = CFG.BATCH_SIZE
    split_ratio = CFG.SPLIT_RATIO
    learning_rate = CFG.LEARNING_RATE

    def __init__(self, parser):
        self.model_type = parser.model_type
        self.csv_path = parser.csv_path

    def load_data(self):
        inputs = []
        outputs = []

        df = pd.read_csv(self.csv_path)
        for idx, row in df.iterrows():
            inputs.append(row['img_path'])

            if self.model_type == 'age_gender':
                outputs.append((row['gender'], row['age']))
            elif self.model_type == 'age_gender_emotion':
                outputs.append((row['gender'], row['age'], row['emotion']))

        return inputs, outputs

    def main(self):
        if self.model_type == 'age_gender':
            model = MTLAgeGender().model()

            loss = {'gender': 'binary_crossentropy',
                    'age': 'categorical_crossentropy'}
            metrics = ['acc']
            opt = Adam(lr=self.learning_rate)

            age_drawer = DrawGraph(self.model_type, 'age')
            gender_drawer = DrawGraph(self.model_type, 'gender')
            early_stop_check = MultiOutputEarlyStoppingAndCheckpoint(['val_gender_acc', 'val_age_acc'],
                                                                     self.model_type, 3)
            callbacks = [age_drawer, gender_drawer, early_stop_check]

        elif self.model_type == 'age_gender_emotion':
            model = MTLAgeGenderEmotion().model()

            loss = {'gender': 'binary_crossentropy',
                    'age': 'categorical_crossentropy',
                    'emotion': 'categorical_crossentropy'}
            metrics = ['acc']
            opt = Adam(lr=self.learning_rate)

            age_drawer = DrawGraph(self.model_type, 'age')
            gender_drawer = DrawGraph(self.model_type, 'gender')
            emotion_drawer = DrawGraph(self.model_type, 'emotion')
            early_stop_check = MultiOutputEarlyStoppingAndCheckpoint(['val_gender_acc', 'val_age_acc',
                                                                      'val_emotion_acc'],
                                                                     self.model_type, 5)
            callbacks = [age_drawer, gender_drawer, emotion_drawer, early_stop_check]

        else:
            return None

        x_train, y_train = self.load_data()
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=self.split_ratio)

        train_data = DataGenerator(x_train, y_train, self.model_type, self.batch_size, shuffle=True, augment=True)
        val_data = DataGenerator(x_test, y_test, self.model_type, self.batch_size)

        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)

        _ = model.fit_generator(generator=train_data,
                                validation_data=val_data,
                                epochs=200,
                                steps_per_epoch=len(train_data),
                                validation_steps=len(val_data),
                                callbacks=callbacks,
                                verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--csv_path', required=True, type=str)
    parser = parser.parse_args()

    if parser.model_type in ['age_gender', 'age_gender_emotion']:
        create_folders(parser.model_type)
        train = Training(parser)
        train.main()

    else:
        print('Age_gender and age_gender_emotion are available!')
