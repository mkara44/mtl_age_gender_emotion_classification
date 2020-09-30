# Training Configuration
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
SPLIT_RATIO = 0.2
INPUT_SHAPE = 64
PRETRAINED_AGE_GENDER_MODEL_PATH = './pretrained_models/age_gender/age_gender_epoch22_age0.674_gender0.927.hdf5'

# Pretrained Models' Label Hash
GENDER_LABEL_HASH = {'Male': 1, 'Female': 0}

AGE_LABEL_HASH = {'Child': [1, 0, 0, 0, 0], 'Young': [0, 1, 0, 0, 0],
                  'Middle Age': [0, 0, 1, 0, 0], 'Old': [0, 0, 0, 1, 0],
                  'Very Old': [0, 0, 0, 0, 1]}

EMOTION_LABEL_HASH = {"Angry": [1, 0, 0, 0, 0, 0, 0], "Disgust": [0, 1, 0, 0, 0, 0, 0],
                      "Fear": [0, 0, 1, 0, 0, 0, 0], "Happy": [0, 0, 0, 1, 0, 0, 0],
                      "Sad": [0, 0, 0, 0, 1, 0, 0], "Surprise": [0, 0, 0, 0, 0, 1, 0],
                      "Neutral": [0, 0, 0, 0, 0, 0, 1]}
