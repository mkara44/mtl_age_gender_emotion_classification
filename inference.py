import cv2
import dlib
import argparse
import numpy as np
from keras.models import load_model

# Additional Scripts
import config as CFG

model = None
face_detector = dlib.get_frontal_face_detector()

import tensorflow as tf
from keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
set_session(sess)


def preprocess(gray, coord):
    x, y, w, h = coord

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    gray = gray[y1:y2, x1:x2]
    gray = cv2.resize(gray, (CFG.INPUT_SHAPE, CFG.INPUT_SHAPE))
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    gray = gray / 255.0

    return gray


def postprocess(predictions):
    if len(predictions) == 2:
        gender_prob = predictions[0][0][0]
        age_prob = predictions[1][0]

        gender_label = list(CFG.GENDER_LABEL_HASH.keys())[0] if gender_prob >= 0.5 else \
            list(CFG.GENDER_LABEL_HASH.keys())[1]
        age_label = list(CFG.AGE_LABEL_HASH.keys())[np.argmax(age_prob)]

        results = [gender_label, age_label]

    elif len(predictions) == 3:
        gender_prob = predictions[0][0][0]
        age_prob = predictions[1][0]
        emo_prob = predictions[2][0]

        gender_label = list(CFG.GENDER_LABEL_HASH.keys())[0] if gender_prob >= 0.5 else \
            list(CFG.GENDER_LABEL_HASH.keys())[1]
        age_label = list(CFG.AGE_LABEL_HASH.keys())[np.argmax(age_prob)]
        emotion_label = list(CFG.EMOTION_LABEL_HASH.keys())[np.argmax(emo_prob)]

        results = [gender_label, age_label, emotion_label]

    else:
        results = None

    return results


def find_faces(gray):
    all_faces = []

    faces = face_detector(gray)
    for face in faces:
        x = int(face.left()) if int(face.left()) > 0 else 0
        y = int(face.top()) if int(face.top()) > 0 else 0
        w = int(face.width()) if int(face.width()) > 0 else 0
        h = int(face.height()) if int(face.height()) > 0 else 0

        all_faces.append([x, y, w, h])

    return all_faces


def model_pipe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coord = find_faces(gray)
    for coord in face_coord:
        frame_to_model = preprocess(gray, coord)
        predictions = model.predict(frame_to_model)
        results = postprocess(predictions)

        if results is not None:
            cv2.putText(frame, ', '.join(results), (coord[0], coord[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]),
                          (0, 0, 255), 2)


def run_model_w_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model_pipe(frame)

        cv2.imshow('Frame', frame)
        pk = cv2.waitKey(2)
        if pk == ord('q'):
            break

    cv2.destroyAllWindows()


def run_model_w_image(image_path):
    frame = cv2.imread(image_path)

    model_pipe(frame)
    cv2.imshow('Frame', frame)
    cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--source', choices=['image', 'video'], required=True, type=str)
    parser.add_argument('--path')
    parser = parser.parse_args()

    model = load_model(parser.model_path)

    if parser.source == 'image':
        run_model_w_image(parser.path)
    elif parser.source == 'video':
        run_model_w_video(parser.path if parser.path != '0' else 0)
