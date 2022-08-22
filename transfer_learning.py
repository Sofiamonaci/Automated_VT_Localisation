"""

Pre- and/or post processing functions for the AI automated VT localisation algorithm + training/testing

This module provides methods to read/upload .mat files (from matlab), prepare training and testing datasets,
re-training model (transfer learning) and testing such model

"""

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, mean_absolute_error


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    :param filename:    (str) filename of .mat file to open

    :return:            dictionary
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries

    :param dict:    (dict) dictonary to check keys of

    :return:        (dict) dictonary
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries

    :param matobj:      matlab object/structure
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def read_training(filename, leads=16, flag_weights=True, flag_norm=True):
    """
    return tensor from mat file to use as training for AI architecture

    :param filename:        (str) .mat filename
    :param leads:           (int)  number of leads (default = 16)
    :param flag_weights:    (bool)  weighting classes if they are unbalanced (default = True)
    :param flag_norm:       (bool)  normalising signals according to lead with highest absolute voltage

    :return:            data_train, data_test, label_train, label_test, ground_test, ground_train
    """

    print('Reading %s ...' % (filename))
    # Read data from mat file
    data = loadmat(filename)
    # Construct dictionary
    dict = _check_keys(data)
    # Check keys
    field_dict = [a for a in dict if '__' not in a]

    # Checking if dictionaru has training dataset/labels
    if all(elem in field_dict for elem in ['data_train','label_train']):
        print('\nData read contains training dataset and labels ...')
        data_train = data['data_train']
        label_train = data['label_train']

        ########## Preparing training datasets for training ##########

        # Reducing number of leads
        if leads<16:
            data_train = data_train[:, :, :leads]

        n_pace = data_train.shape[0]
        n_time = data_train.shape[1]

        print('Shape training data: (%d,%d,%d)' % (data_train.shape[0], data_train.shape[1], data_train.shape[2]))

        # Weighting PHI classes if flag_weights is True
        if flag_weights:

            # Compute label_train PHI counts and corresponding weights
            label_count = np.bincount(np.argmax(label_train['PHI'], axis=1))
            weights = (1 / label_count) * (np.sum(label_count) / 2.0)

            # Plotting PHI distribution and corresponding class weights
            plt.subplot(2, 1, 1)
            plt.hist(np.argmax(label_train['PHI'], axis=1))
            plt.title('PHI distribution')
            plt.subplot(2, 1, 2)
            plt.hist(weights, len(weights))
            plt.title('PHI weights')
            plt.show()

        if flag_norm:
            print('\nNormalising training data according to lead with highest absolute voltage ...')
            new_data = np.zeros((n_pace, n_time, leads))
            for i in range(n_pace):
                l_max = np.max(abs(data_train[i, :, :]), axis=0).argmax()
                new_data[i, :, :] = data_train[i, :, :] / max(abs(data_train[i, :, l_max]))

            data_train = new_data

        return data_train, label_train

def read_testing(filename, leads=16, flag_weights=True, flag_norm=True):

        """
        return tensor from mat file to use as testing for AI architecture

        :param filename:        (str) .mat filename
        :param leads:           (int)  number of leads (default = 16)
        :param flag_weights:    (bool)  weighting classes if they are unbalanced (default = True)
        :param flag_norm:       (bool)  normalising signals according to lead with highest absolute voltage

        :return:            data_train, data_test, label_train, label_test, ground_test, ground_train
        """

        print('Reading %s ...' % (filename))
        # Read data from mat file
        data = loadmat(filename)
        # Construct dictionary
        dict = _check_keys(data)
        # Check keys
        field_dict = [a for a in dict if '__' not in a]
        # Checking if .mat file also have testing data
        if all(elem in field_dict for elem in ['data_test', 'label_test']):

            print('\nData read contains testing dataset and labels ...')
            data_test = data['data_test']
            label_test = data['label_test']

            # Reducing number of leads
            if leads < 16:
                data_test = data_test[:, :, :leads]

            # Checking if data shape is not a tensor
            if len(data_test.shape) < 3:
                data_test = np.reshape(data_test, [1, data_test.shape[0], leads])

            # Normalising testing data if flag_norm is true
            if flag_norm:
                print('Normalising data train according to lead with highest absolute voltage')
                new_data = data_test * 0
                for i in range(data_test.shape[0]):
                    l_max = np.max(abs(data_test[i, :, :]), axis=0).argmax()
                    new_data[i, :, :] = data_test[i, :, :] / max(abs(data_test[i, :, l_max]))

                data_test = new_data

            print('Shape testing data: (%d,%d,%d)' % (data_test.shape[0], data_test.shape[1], data_test.shape[2]))

        return data_test, label_test

def transfer_learning_training(data_train,
                      label_train,
                      model,
                      output_name,
                      flag = 'phi',
                      class_weights = [],
                      params={'learning_rate': 0.00005,
                              'loss': "categorical_crossentropy",
                              'metrics': ["accuracy"],
                              'batch_size': 50,
                              'epochs': 250}):

    """
    re-train trained model according to data provided (PHI and Z_RHO architecture). This version uses early stopping with patience=5 and restore_best_weights=True
    
    :param data_train:      (np.ndarray: 3 dim)  training dataset
    :param train_label:     (np.ndarray: 3 dim)  training label
    :param model:           (tensorflow model) model to load
    :param output_name:     (str) output name of where to save model
    :param flag:            (str) 'phi' or 'z_rho'
    :param class_weights:    Class weights (if empty, no class weighting is performed)
    :param params:          (dict) dictionary containing parameters 'learning_rate', 'metrics', 'loss', 'batch_size', 'epochs' (defaults here are for PHI architecture)

    """

    # Freezing top 7 layers
    print('\n\n Setting up Transfer Learning between Pacing and Scar-related VTs ...\n\n')
    print('--------------------------------------------------------------------------')
    # Freezing CNN (top layers)
    for layer in model.layers[:7]:
        print(layer)
        print('Before: ' + str(layer.trainable))
        layer.trainable = False
        print('After: ' + str(layer.trainable))

    print('--------------------------------------------------------------------------')

    # Initialising parameters
    learning_rate = params['learning_rate']
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=params['loss'], optimizer=optimizer, metrics=params['metrics'])

    # Add early stopping
    callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    print('Saving training log in csv file: {}'.format(output_name + '_training.log'))
    csv_logger = CSVLogger(output_name + '_training.log', separator=',', append=False)

    # Re-train PHI classification
    if not class_weights:
        print('Not weighting classes ...')
        re_training = model.fit(data_train, label_train[flag.upper()], validation_split=0.1,
                                        batch_size=params['batch_size'], epochs=params['epochs'], verbose=1,
                                        callbacks=[callback, csv_logger])
    else:
        re_training = model.fit(data_train, label_train[flag.upper()], validation_split=0.1,
                                        batch_size=params['batch_size'], epochs=params['epochs'], verbose=1,
                                        callbacks=[callback, csv_logger], class_weight=class_weights)

    print('Saving model in %s..\n\n' % (output_name))
    tf.keras.models.save_model(model, output_name, save_format='tf')

    return re_training, model

def plot_training_validation_curves(training, output_figure, metrics):

    """
    plotting training/validation curves

    :param training:        (tf trained model history) training history (as returned from model.fit above)
    :param output_figure:   (str) name for output figure
    :param metrics:         (str) 'mse' or 'accuracy'

    :return                 training_history, trained_model
    """


    loss_train = np.asarray(training.history['loss'])
    loss_test = np.asarray(training.history['val_loss'])

    acc_train = np.asarray(training.history[metrics.lower()])
    acc_test = np.asarray(training.history['val_' + metrics.lower()])

    # Plotting

    fig = plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.plot(loss_train, '-b', label='Train')
    ax.plot(loss_test, '-r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax.legend()

    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(acc_train, '-b',
             label='Train ' + metrics.upper() + ': ' + str(np.round(np.mean(acc_train), 4)) + ' +/- ' + str(np.round(np.std(acc_train), 4)))
    ax1.plot(acc_test, '-r', label='Validation '+ metrics.upper() + ': ' + str(np.round(np.mean(acc_test), 4)) + ' +/- ' + str(
        np.round(np.std(acc_test), 4)))
    plt.xlabel('Epochs')
    plt.ylabel(metrics.upper)
    ax1.legend()

    print('Saving ... %s' % (output_figure))
    fig.savefig(output_figure)

def transfer_learning_testing(data_test,
                              label_test,
                              model,
                              outfile,
                              flag='phi',
                              n_time = 150,
                              start_vt = list(range(30,200,10))):

    """
    testing re-trained model (either phi or z-rho according to flag provided)

    :param data_test:       (np.ndarray: 3 dims) testing dataset
    :param label_test:      (np.ndarray) testing labels
    :param model:           (tf trained model) trained model to test
    :param outfile:         (str) output file for predicted values
    :param flag:            (str) 'phi' or 'z_rho'
    :param n_time:          (int) n_steps of signals (default: 150)
    :param start_vt:        (list) list of integers to consider different cropped windows of testing dataset
    :
    """

    print('\n\n Testing \n\n')
    original_data = data_test

    for a in start_vt:

        end_vt = a + n_time
        data_test = original_data[:, a:end_vt, :]

        print(data_test.shape)

        # Test model
        predicted = model.predict(data_test)
        # Save output probabilities

        if 'phi' in flag.lower():
                testing_score = accuracy_score(np.argmax(predicted, axis=1),
                                               np.argmax(label_test, axis=1)) * 100
                print('Accuracy score: %.3f\n' % (testing_score))

        else:
                testing_score = mean_absolute_error(predicted, label_test, multioutput='raw_values')
                print('Neg mean absolute error: [%.3f,%.3f]\n' % (
                testing_score[0], testing_score[1]))


        print('\n\nPrinting out predicted values in %s\n\n' % (outfile))
        np.savetxt(outfile, predicted)





print(list(range(30,200,10)))

