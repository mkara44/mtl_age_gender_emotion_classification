import os


def create_folders(model_type):
    if not os.path.exists('./pretrained_models'):
        os.mkdir('./pretrained_models')

    if not os.path.exists(f'./pretrained_models/{model_type}'):
        os.mkdir(f'./pretrained_models/{model_type}')

    if not os.path.exists('./graphs'):
        os.mkdir('./graphs')

    if not os.path.exists(f'./graphs/{model_type}'):
        os.mkdir(f'./graphs/{model_type}')
