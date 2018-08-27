import os
import threading

from os import path

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from audio2video import create_model
from audio2video import BatchGenerator


SEQUENCE_LENGTH = 100
N_AUDIO_FEATURES = 28
N_MOUTH_FEATURES = 20
BATCH_SIZE = 50
STEPS_PER_EPOCH = 10
NUMBER_OF_EPOCHS = 10
ROOT_DIR = path.dirname(path.abspath(__file__))
LOG_DIR = path.join(ROOT_DIR, 'tensorboard')
MODEL_WEIGHTS_FILE = path.join(ROOT_DIR, 'weights.hdf5')
DATA_PATH = path.join(ROOT_DIR, 'data', 'obama')


def launch_tensorboard(log_dir):
    os.system(f"tensorboard --logdir {log_dir}")


threading.Thread(target=launch_tensorboard,
                 args=(LOG_DIR,)).start()


batch_generator = \
    BatchGenerator(DATA_PATH,
                   batch_size=BATCH_SIZE,
                   sequence_length=SEQUENCE_LENGTH,
                   normalize_input=True,
                   normalize_output=False,
                   shuffle=True)


model = create_model()
if path.exists(MODEL_WEIGHTS_FILE):
    print(f"loading weights from {MODEL_WEIGHTS_FILE}")
    model.load_weights(MODEL_WEIGHTS_FILE)

tensor_board = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=False
)
checkpoint = ModelCheckpoint(MODEL_WEIGHTS_FILE,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

history = model.fit_generator(
    batch_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=NUMBER_OF_EPOCHS,
    verbose=1,
    max_queue_size=10,
    use_multiprocessing=True,
    shuffle=True,
    callbacks=[tensor_board, checkpoint]
)
