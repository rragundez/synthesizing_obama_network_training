"""Deep learning model to map audio features to mouth features
"""
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

SEQUENCE_LENGTH = 100
N_AUDIO_FEATURES = 28
N_MOUTH_FEATURES = 20
LEARNING_RATE = 0.001
DECAY_RATE = 0.0
GRADIENT_NORM_CLIP_VALUE = 10


def create_model(rnn_hidden_neurons=60, rnn_dropout_prob=0.0):

    model = Sequential()
    model.add(
        LSTM(rnn_hidden_neurons,
             input_shape=(SEQUENCE_LENGTH, N_AUDIO_FEATURES),
             activation='tanh',
             recurrent_activation='hard_sigmoid',
             use_bias=True,
             dropout=rnn_dropout_prob,
             recurrent_dropout=rnn_dropout_prob,
             return_sequences=True)
    )
    model.add(
        TimeDistributed(
            Dense(N_MOUTH_FEATURES,
                  activation='linear',
                  use_bias=True)
        )
    )

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(lr=LEARNING_RATE,
                       decay=DECAY_RATE,
                       clipnorm=GRADIENT_NORM_CLIP_VALUE))
    print(model.summary())
    return model
