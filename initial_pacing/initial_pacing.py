"""

Training initial pacing

This module provides methods to build the AI model for automating focal and scar-related VT localisation

"""

import tensorflow as tf
from AI_architecture import build_model
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger


def train_model(data,
                labels,
                out_model,
                batch_size = 100,
                epochs = 50,
                class_weights = [],
                params={'dropout': 0.2,
                        'kernel': 5,
                        'learning_rate': 0.0002,
                        'dim': 16,
                        'feat': 2,
                        'loss': 'mean_absolute_error',
                        'metrics': 'mse',
                        'activation': 'linear'}):

    '''
    Training initial pacing


    :param data:            (tensor) training dataset
    :param labels:          (tensor) labels for training
    :param out_model:       (str)    path name of output model
    :param batch_size:      (int) batch size
    :param epochs:          (int) number of epochs
    :param class_weights:   (np.ndarray) class weights (default: empty list)
    :param params:          (dict)   'dropout' dropout rate after LSTMs. (default: 0.2 for Z_RHO regression, similar for PHI classification)
                                        'kernel' size of convolution kernel (default: 5 for both Z_RHO and PHI)
                                        'learning_rate' learning rate (default: 0.0002 for PHI, 0.0001 for Z_RHO)
                                        'dim' number of leads
                                        'feat' number of features, either 2 for regression (default) or 17 for PHI classification
                                        'loss' (either 'mean_absolute_error' or 'categorical_crossentropy')
                                        'metrics' (either 'mse' or 'accuracy')
                                        'activation' (either 'linear' or 'softmax'

    :return:                training (history), model
    '''

    print('Training initial pacing model and saving into: %s\n\n' % (out_model))

    # Building model
    model = build_model(params)
    # Creating Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
    # Creating logger for saving training history
    print('Saving training log in csv file: {}'.format(out_model + '_training.log'))
    csv_logger = CSVLogger(out_model + '_training.log', separator=',', append=False)

    # Randonly stratified data into training and validation (90%-10% split)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=5)
    for train_index, test_index in sss.split(data, labels):
        train_data, val_data = data[train_index, :, :], data[test_index, :, :]
        train_label, val_label = labels[train_index, :], labels[test_index, :]


    # Training model according to weighted classes
    if not class_weights:
        training = model.fit(train_data, train_label, validation_data=(val_data, val_label),
                             batch_size=batch_size, epochs=epochs,
                             verbose=1, callbacks=[callback, csv_logger])
    else:
        training = model.fit(train_data, train_label, validation_data=(val_data, val_label),
                             batch_size=params['batch_size'], epochs=epochs,
                             verbose=1, callbacks=[callback, csv_logger],
                             class_weight=class_weights)

    # Save model
    tf.keras.models.save_model(model, out_model, save_format='tf')

    return training, model