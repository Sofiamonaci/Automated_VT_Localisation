"""
Test script to re-trained AI architecture on ECGs/EGMs of scar-related VTs and test it on unseen signals
"""

import tensorflow as tf
from transfer_learning import generate_model_name, read_training, read_testing, transfer_learning_training,plot_training_validation_curves, transfer_learning_testing
import os

'''
INITIALISE ALL VARIABLES
'''

feat = 17
VT_episodes = '500'
timepoints = 150
type_sig = 'ECG'
leads = 16
flag_weights = True
flag_norm = True
batch_size_phi = 100 # n_batch
epochs_phi = 50
batch_size_reg = 75
epochs_reg = 70
lr_phi = 0.0002
lr_reg = 0.0001
vt_ext = '19N_OE_NK_QR_EA_'             # Prefix of VT training, based on VT training datasets
pacing_ext = 'Norm_19N_EA_QR_NK_OE_'    # Prefix of pacing training, based on training datasets (torso models)
VT_params_phi = {'batch_size': 50, 'epochs': 250, 'learning_rate': 0.00005, 'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}                      # Parameters for transfer learning PHI
VT_params_reg = {'batch_size': 50, 'epochs': 250, 'learning_rate': 0.0001,  'loss': 'mean_absolute_error',      'metrics': ['mse']}                      # Parameters for transfer learning Z_RHO

# VT datasets namefiles | training and testing
re_training_dataset = ''.join(['../VT_datasets/{}{}_noise_sc_windows_data_{}_{}timepoints.mat'.format(vt_ext,VT_episodes,type_sig.lower(),timepoints)])
testing_name = 'SBRT1_VT_test_data.mat'
testing_dataset     = '../VT_datasets/' + testing_name
start_vt = list(range(0,20,10))

'''
LOADING TRAINED MODELS (FROM INITIAL PACING)
'''

# ../../initial_pacing/trained_models'
folder_phi = '../trained_models/'
folder_reg = '../trained_models/'

model_phi_name = generate_model_name(folder = folder_phi + pacing_ext, leads=leads, timepoints= timepoints, feat=17, n_batch=batch_size_phi, n_epochs=epochs_phi)
model_reg_name = generate_model_name(folder = folder_reg + pacing_ext, leads=leads, timepoints= timepoints,  feat=2, n_batch=batch_size_reg, n_epochs=epochs_reg)

print('Loading %s ..\n\t %s ...\n\n' %(model_phi_name,model_reg_name))

pacing_model_phi = tf.keras.models.load_model(model_phi_name)
pacing_model_reg = tf.keras.models.load_model(model_reg_name)

print('Batch size: %s Epochs: %s for trained PHI Classification model (pacing training)\n'%(batch_size_phi,epochs_phi))
print('Batch size: %s Epochs: %s for trained Z and RHO Regression model (pacing training)\n'%(batch_size_reg,epochs_reg))

'''
OUTPUT FOLDERS OF RE_TRAINED MODELS (SCAR-RELATED VTs)
'''

# The folders where re-trained models are saved contain the information from the pacing trained models. The actual name
# of the re-trained models have then the information of the re-training

name = ['PHI','Z_RHO']
out_folder = {'PHI': '', 'Z_RHO': ''}

out_folder['PHI'] = '../trained_models/pacing_training_' + pacing_ext + 'lr'+str(lr_phi)+'_epochs'+str(epochs_phi)+'_batchsize'+str(batch_size_phi)+'_leads'+str(leads) + '_earlystop_7layers_'+ str(feat)+ 'PHI/'
out_folder['Z_RHO'] = '../trained_models/pacing_training_' + pacing_ext + 'lr'+str(lr_reg)+'_epochs'+str(epochs_reg)+'_batchsize'+str(batch_size_reg)+'_leads'+str(leads) + '_earlystop_7layers_Regression/'

# Creating folders if they do not exist
for i in name:

    if not os.path.exists(out_folder[i]):
        print('Creating %s folder \n\n' % (out_folder[i]))
        os.mkdir(out_folder[i])

'''
GENERATE TRANSFER LEARNING MODEL NAMES (to LOAD or WRITE)
'''

re_trained_model_phi_name =  out_folder['PHI']  + flag_norm*'Norm_' + flag_weights*'Class_weights_' + 'VT_training_' + vt_ext + 'lr'+str(VT_params_phi['learning_rate'])+'_epochs'+str(VT_params_phi['epochs'])+'_batchsize'+str(VT_params_phi['batch_size']) + '_' + str(feat)+ 'PHI'
re_trained_model_reg_name = out_folder['Z_RHO'] + flag_norm*'Norm_' + flag_weights*'Class_weights_' + 'VT_training_' + vt_ext + 'lr'+str(VT_params_reg['learning_rate'])+'_epochs'+str(VT_params_reg['epochs'])+'_batchsize'+str(VT_params_reg['batch_size']) + '_Regression'

print('Loading %s ..\n\t %s ...\n\n' %(re_trained_model_phi_name,re_trained_model_reg_name))

# if model names already exist, models are used for TESTING and are NOT RE-TRAINED
if not os.path.exists(re_trained_model_phi_name):
    print('Re-training model: {} ...\n'.format(re_trained_model_phi_name))
    flag_re_training_phi = True
else:
    print('Loading model: {} ...\n'.format(re_trained_model_phi_name))
    flag_re_training_phi = False
    re_trained_model_phi = tf.keras.models.load_model(re_trained_model_phi_name)

if not os.path.exists(re_trained_model_reg_name):
    print('Re-training model: {} ...\n'.format(re_trained_model_reg_name))
    flag_re_training_reg = True
else:
    print('Loading model: {} ...\n'.format(re_trained_model_reg_name))
    flag_re_training_reg = False
    re_trained_model_reg = tf.keras.models.load_model(re_trained_model_reg_name)


'''
RE-TRAINING if FLAGS are TRUE
'''

if flag_re_training_phi:
    print('\n\nRE-TRAINING PHI CLASSIFICATION ON VT EPISODES\n\n')
    print('First, loading training datasets ...')
    data_train, label_train, class_weights_phi = read_training(filename = re_training_dataset, leads=leads, flag_weights=flag_weights, flag_norm=flag_norm)
    # Then, re-trained
    re_training_phi, re_trained_model_phi = transfer_learning_training(data_train,
                               label_train,
                               model = pacing_model_phi,
                               output_name = re_trained_model_phi_name,
                               flag='phi',
                               class_weights=class_weights_phi,
                               params=VT_params_phi)


if not os.path.exists(re_trained_model_phi_name + '_training.png'):
    plot_training_validation_curves(re_trained_model_phi_name + '_training.log', 'accuracy')


if flag_re_training_reg:
    print('\n\nRE-TRAINING Z_RHO REGRESSION ON VT EPISODES\n\n')
    print('First, loading training datasets ...')
    data_train, label_train, class_weights_reg = read_training(filename = re_training_dataset, leads=leads, flag_weights=flag_weights, flag_norm=flag_norm)
    # Then, re-trained
    re_training_reg, re_trained_model_reg = transfer_learning_training(data_train,
                               label_train,
                               model = pacing_model_reg,
                               output_name = re_trained_model_reg_name,
                               flag='z_rho',
                               class_weights=class_weights_reg,
                               params=VT_params_reg)


if not os.path.exists(re_trained_model_reg_name + '_training.png'):
    plot_training_validation_curves(re_trained_model_reg_name + '_training.log', 'mse')


'''
TESTING
'''

testing_phi_outfile = re_trained_model_phi_name + '/testing/'
testing_reg_outfile = re_trained_model_reg_name + '/testing/'

# Creating output testing folders if already not existing
if not os.path.exists(testing_phi_outfile):
    print('Creating %s folder \n\n' % (testing_phi_outfile))
    os.mkdir(testing_phi_outfile)

testing_phi_outfile += testing_name[:-4] + '_PHI_'

if not os.path.exists(testing_reg_outfile):
    print('Creating %s folder \n\n' % (testing_reg_outfile))
    os.mkdir(testing_reg_outfile)

testing_reg_outfile += testing_name[:-4] + '_Z_RHO_'

# Loading testing datasets
print('\n\nTESTING\n\n')
print('First, loading testing dataset ...')
data_test, label_test = read_testing(filename=testing_dataset, leads=leads, flag_norm=flag_norm)
# Testing PHI first
print('Predicting PHI classes ...')
transfer_learning_testing(data_test,
                            label_test,
                              model = re_trained_model_phi,
                              outfile = testing_phi_outfile,
                              flag='phi',
                              n_time = timepoints,
                              start_vt = start_vt)
# Testing Z_RHO regression
print('Predicting Z_RHO ...')
transfer_learning_testing(data_test,
                            label_test,
                              model = re_trained_model_reg,
                              outfile = testing_reg_outfile,
                              flag='z_rho',
                              n_time = timepoints,
                              start_vt = start_vt)