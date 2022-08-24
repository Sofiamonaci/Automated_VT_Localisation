"""
Test script to train AI architecture to locate focal paced beats in UVCs
"""


from intial_pacing import train_model
from transfer_learning import read_training, plot_training_validation_curves
import os
'''
INITIALISE ALL VARIABLES
'''

leads = 16
timepoints = 150
type_sig = 'ECG'
flag_weights = False
flag_norm = True
batch_size_phi = 100
epochs_phi = 50
batch_size_reg = 75
epochs_reg = 70
model_ext = '19N_EA_QR_NK_OE_'


# Parameters for phi classification
params_phi  =   {'dropout': 0.2,
                'kernel': 5,
                'learning_rate': 0.0002,
                'dim': leads,
                'feat': 17,
                'loss': 'categorical_crossentropy',
                'metrics': 'accuracy',
                'activation': 'softmax'}

# Parameters for z and rho regression
params_reg  =   {'dropout': 0.2,
                'kernel': 5,
                'learning_rate': 0.0002,
                'dim': leads,
                'feat': 2,
                'loss': 'mean_absolute_error',
                'metrics': 'mse',
                'activation': 'linear'}

# pacing datasets namefiles
training_dataset = ''.join(['../VT_datasets/trainingCNN1D_{}_5noise_{}leads_3windows_{}timepoints.mat'.format(type_sig,leads,timepoints)])


'''
GENERATE OUTPUT MODEL NAMES for training  
'''

name = ['PHI','Z_RHO']
out_folder = {'PHI': '', 'Z_RHO': ''}

out_folder['PHI'] = '../trained_models/'
out_folder['Z_RHO'] = '../trained_models/'


model_phi_name =  out_folder['PHI']  + flag_norm*'Norm_' + flag_weights*'Class_weights_' + model_ext + \
                  type_sig.upper() + '_' + str(leads) + 'leads_17PHI_' + str(timepoints) \
                  + 'timepoints_LSTMS128_64_attention_' + str(batch_size_phi)+'batch_'+ \
                  str(epochs_phi) +'epochs_earlystop/'

model_reg_name = out_folder['Z_RHO'] + flag_norm*'Norm_' + flag_weights*'Class_weights_' + model_ext + \
                 type_sig.upper() + '_' + str(leads) + 'leads_Regression_' + str(timepoints) \
                  + 'timepoints_LSTMS128_64_attention_' + str(batch_size_reg)+'batch_'+ \
                  str(epochs_reg) +'epochs_earlystop/'


# if model names already exist, models are not re-trained!
if not os.path.exists(model_phi_name):
    print('Training model: {} ...\n'.format(model_phi_name))
    flag_training_phi = True
else:
    print('Model {} already exists!\n'.format(model_phi_name))
    flag_training_phi = False

if not os.path.exists(model_reg_name):
    print('Training model: {} ...\n'.format(model_reg_name))
    flag_training_reg = True
else:
    print('Model {} already exists!\n'.format(model_reg_name))
    flag_training_reg = False


'''
TRAINING if FOLDERS ARE NOT ALREADY PRESENT
'''

# Loading training data and labels
print('Loading training datasets ...')
data_train, label_train, class_weights_phi = read_training(filename = training_dataset, leads=leads, flag_weights=flag_weights, flag_norm=flag_norm)

if flag_training_phi:
    print('\n\nTRAINING PHI CLASSIFICATION ON FOCAL BEATS\n\n')
    # Train
    _, model_phi = train_model(data_train,
                label_train,
                model_phi_name,
                batch_size = batch_size_phi,
                epochs = epochs_phi,
                class_weights = class_weights_phi,
                params = params_phi)

if not os.path.exists(model_phi_name + '_training.png'):
    plot_training_validation_curves(model_phi_name + '_training.log', 'accuracy')


if flag_training_reg:
    print('\n\nTRAINING Z_RHO REGRESSION ON FOCAL BEATS\n\n')
    # Train
    _, _model_reg = train_model(data_train,
                label_train,
                model_reg_name,
                batch_size = batch_size_reg,
                epochs = epochs_reg,
                class_weights = [],
                params = params_reg)

if not os.path.exists(model_reg_name + '_training.png'):
    plot_training_validation_curves(model_reg_name + '_training.log', 'mse')