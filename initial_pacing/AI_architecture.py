"""

AI architecture incorporating CNN + LSTMs + attention mechanisms

This module provides methods to build the AI model for automating focal and scar-related VT localisation

"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from tensorflow.keras.layers import Layer

from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam


# Cite the following for these functions
# @INPROCEEDINGS{8461990,
#author={G. I. Winata and O. P. Kampman and P. Fung},
#booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#title={Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision},
#year={2018},
#volume={},
#number={},
#pages={6204-6208},
#doi={10.1109/ICASSP.2018.8461990},
#ISSN={2379-190X},
#month={April},}

def dot_product(x, kernel):
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights
	Returns:
	"""
	if K.backend() == 'tensorflow':
		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
		return K.dot(x, kernel)


class AttentionWithContext(Layer):

    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    follows these equations:

    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.

    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                             initializer=self.init,
                             name='{}_u'.format(self.name),
                             regularizer=self.u_regularizer,
                             constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return weighted_input

        def compute_output_shape(self, input_shape):
            return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights

    follows this equation:

    (1) v = \sum_t(\alpha_t * h_t)

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#####################################################################################################

def build_model(params = {'dropout': 0.2,
                          'kernel': 5,
                          'learning_rate': 0.0002,
                          'dim': 16,
                          'feat': 2,
                          'loss': 'mean_absolute_error',
                          'metrics': 'mse',
                          'activation': 'linear'}):

    '''
    Building AI architecture made of a deep 1D CNN, 2 LSTM cells and an attention mechanisms, Arguments of this
    function depend on whether we are classifying PHI or finding Z and RHO regression values. Default values are for the latter

    :param params:             (dict)   'dropout' dropout rate after LSTMs. (default: 0.2 for Z_RHO regression, similar for PHI classification)
                                        'kernel' size of convolution kernel (default: 5 for both Z_RHO and PHI)
                                        'learning_rate' learning rate (default: 0.0002 for PHI, 0.0001 for Z_RHO)
                                        'dim' number of leads
                                        'feat' number of features, either 2 for regression (default) or 17 for PHI classification
                                        'loss' (either 'mean_absolute_error' or 'categorical_crossentropy')
                                        'metrics' (either 'mse' or 'accuracy')
                                        'activation' (either 'linear' or 'softmax'
    :return:                    model
    '''

    # 1D CNN
    model = Sequential()
    model.add(Conv1D(64, params['kernel'], padding="same", input_shape=(None, params['dim']), activation='relu'))
    model.add(MaxPooling1D(2, strides=2))  # original first three 2,2

    model.add(Conv1D(128, params['kernel'], padding="same", activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(MaxPooling1D(1, strides=1))

    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(Conv1D(256, 3, padding="same", activation='relu'))
    model.add(MaxPooling1D(1, strides=1))

    # 2 LSTM cells
    model.add(LSTM(128, activation='sigmoid', input_shape=(None, params['dim']), return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(64, activation='sigmoid', input_shape=(None, params['dim']), return_sequences=True))
    model.add(Dropout(params['dropout']))

    # Attention mechanism
    model.add(AttentionWithContext())
    model.add(Addition())

    model.add(Dense(params['feat'], activation=params['activation']))

    # Either Regression or Classification
    optimizer = Adam(lr=params['learning_rate'])
    model.compile(loss=params['loss'], optimizer=optimizer, metrics=params['metrics'])

    return model



