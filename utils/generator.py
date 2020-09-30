import cv2
import random
import numpy as np
from keras.utils import Sequence

# Additional Scripts
import config as CFG


class DataGenerator(Sequence):
    input_shape = CFG.INPUT_SHAPE
    gender_label_hash = CFG.GENDER_LABEL_HASH
    age_label_hash = CFG.AGE_LABEL_HASH
    emotion_hash_label = CFG.EMOTION_LABEL_HASH

    def __init__(self, img_paths, labels, model_type, batch_size=8, shuffle=False, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.model_type = model_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def image_preprocess(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.input_shape, self.input_shape))
        img = np.expand_dims(img, axis=-1)

        return img

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        inputs = []
        gender_outputs = []
        age_outputs = []
        emotion_outputs = []

        for idx in indexes:
            img_paths = self.img_paths[idx]

            if self.model_type == 'age_gender':
                gender, age = self.labels[idx]

            elif self.model_type == 'age_gender_emotion':
                gender, age, emotion = self.labels[idx]
                emotion_outputs.append(self.emotion_hash_label[emotion])

            else:
                return None

            imgs = self.image_preprocess(img_paths)

            inputs.append(imgs)
            gender_outputs.append(self.gender_label_hash[gender])
            age_outputs.append(self.age_label_hash[age])

        inputs = np.array(inputs, dtype="float")
        inputs = inputs / 255.0
        gender_outputs = np.array(gender_outputs, dtype="float")
        age_outputs = np.array(age_outputs, dtype="float")
        outputs = {'gender': gender_outputs, 'age': age_outputs}

        if self.model_type == 'age_gender_emotion':
            emotion_outputs = np.array(emotion_outputs, dtype='float')
            outputs = {'gender': gender_outputs, 'age': age_outputs, 'emotion': emotion_outputs}

        if self.augment:
            inputs = self.augmentation_pipeline(inputs)

        return inputs, outputs

    def augmentation_pipeline(self, inputs):
        selected = []
        possible_augmentations = ['rotate', 'flip']
        for _ in range(random.randint(0, 2)):
            random_idx = random.randint(0, len(possible_augmentations) - 1)
            selected_augmentation = possible_augmentations[random_idx]
            del possible_augmentations[random_idx]

            selected.append(selected_augmentation)

        for aug in selected:
            inputs = self.random_augmentation(aug, inputs)

        return inputs

    def random_augmentation(self, selected, images):
        if selected == 'rotate':
            aug_images = self.rotate(images, [10])
        elif selected == 'flip':
            aug_images = self.flip_horizontal(images)
        else:
            aug_images = images

        return aug_images

    def flip_horizontal(self, inputs):
        flipped_inputs = np.flip(inputs, axis=2)

        return flipped_inputs

    def rotate(self, inputs, rotation_angles):
        rotated_inputs = []

        for angle in rotation_angles:
            angle_list = [angle, -angle]
            angle_idx = random.randint(0, len(angle_list) - 1)
            angle = angle_list[angle_idx]

            M = cv2.getRotationMatrix2D((self.input_shape / 2, self.input_shape / 2), angle, 1.0)

            for idx in range(inputs.shape[0]):
                rotated_input = cv2.warpAffine(inputs[idx], M,
                                               (self.input_shape, self.input_shape),
                                               flags=cv2.INTER_CUBIC)
                rotated_inputs.append(rotated_input)

        return np.reshape(rotated_inputs, (-1, self.input_shape, self.input_shape, 1))
