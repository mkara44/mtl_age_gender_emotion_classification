import warnings
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class DrawGraph(Callback):

    def __init__(self, model_type, output_name):
        super(DrawGraph, self).__init__()
        self.model_type = model_type
        self.output_name = output_name

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get(f'{self.output_name}_loss'))
        self.val_losses.append(logs.get(f'val_{self.output_name}_loss'))
        self.acc.append(logs.get(f'{self.output_name}_acc'))
        self.val_acc.append(logs.get(f'val_{self.output_name}_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label=f"{self.output_name} loss")
        ax1.plot(self.x, self.val_losses, label=f"val {self.output_name} loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label=f"{self.output_name} accuracy")
        ax2.plot(self.x, self.val_acc, label=f"val {self.output_name} accuracy")
        ax2.legend()

        plt.savefig(f'./graphs/{self.model_type}/{self.output_name}')


class MultiOutputEarlyStoppingAndCheckpoint(Callback):
    time = 0
    max_val_acc = {}

    def __init__(self, monitors, model_name, patience):
        super(Callback, self).__init__()
        self.monitors = monitors

        self.patience = patience
        self.model_name = model_name

    def is_increased(self, current_value, monitor_name):
        if self.max_val_acc.get(monitor_name) is None:
            self.max_val_acc[monitor_name] = 0

        passed = False
        if current_value <= self.max_val_acc[monitor_name]:
            print(f'Validation {monitor_name} accuracy'
                  f' did not increase from {round(self.max_val_acc[monitor_name], 5)}')

        else:
            print(f'Validation {monitor_name} accuracy'
                  f' increased from {round(self.max_val_acc[monitor_name], 5)} to {round(current_value, 5)}')

            passed = True
            self.max_val_acc[monitor_name] = current_value

        return passed

    def on_epoch_end(self, epoch, logs={}):
        current_monitor_val_acc = {}
        for monitor in self.monitors:
            current_monitor_val_acc[monitor] = logs.get(monitor)

        if any([True if current_monitor is None else False for current_monitor in current_monitor_val_acc.values()]):
            warnings.warn("Early stopping requires monitors available!", RuntimeWarning)

        print('\n')
        if any([self.is_increased(c_value, m_name) for m_name, c_value in current_monitor_val_acc.items()]):
            acc_to_str = '_'.join([f'{m_name}{round(c_value, 3)}'
                                   for m_name, c_value in current_monitor_val_acc.items()])
            file_path = f'./pretrained_models/{self.model_name}/{self.model_name}_epoch{epoch + 1}' \
                        f'_{acc_to_str}.hdf5'

            self.time = 0
            self.model.save(file_path)
            print(f'Model saved to {file_path}!')

        else:
            self.time += 1
            print('Model did not save!')

        print(f'Early stopping time: {self.time}')
        if self.time == self.patience:
            print('Early stopped!')
            self.model.stop_training = True

        print('\n')
