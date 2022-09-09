
"""
Test script to read predictions from AI, convert them to cartesian and compute localisation errors
"""
import numpy as np
from localisation import Localisation


'''
INITIALISE VARIABLES
'''

uvc_path = '../data/COMBINED_Z_RHO_PHI_V.dat'
mesh_path = '../data/myo_SBRT1.pts'
aha_path = '../data/aha_17segs_SBRT1.pts'
classes_path = '../data/classes_17_SBRT1.dat'
start_vt = [0,10]
file_phi = '../transfer_learning/trained_models/' \
           'pacing_training_Norm_19N_EA_QR_NK_OE_lr0.0002_epochs50_' \
           'batchsize100_leads16_earlystop_7layers_17PHI/' \
           'Norm_Class_weights_VT_training_19N_OE_NK_QR_EA_lr5e-05_epochs250_' \
           'batchsize50_17PHI/testing/' \
           'SBRT1_VT_test_data_PHI_'
file_reg = '../transfer_learning/trained_models/' \
           'pacing_training_Norm_19N_EA_QR_NK_OE_lr0.0001_epochs70_' \
           'batchsize75_leads16_earlystop_7layers_Regression/' \
           'Norm_Class_weights_VT_training_19N_OE_NK_QR_EA_lr0.0001_epochs250_' \
           'batchsize50_Regression/testing/' \
           'SBRT1_VT_test_data_Z_RHO_'

# Ground truth
# file_ground =

# Output folder
out_folder = '../predictions/SBRT1_VT_'

'''
SETTING UP CLASS
'''

loc_class = Localisation(uvc_path = uvc_path,
    mesh_path = mesh_path,
    aha_path = aha_path,
    classes_path = classes_path)

'''
READ PREDICTIONS FOR EACH STARTING VALUE AND COMPUTE CARTESIAN EQUIVALENT(S)
'''

for a in start_vt:
    print('For {} starting ...'.format(a))
    # Convert to Cartesian
    file_phi_a = file_phi + str(a) + 'start.csv'
    file_reg_a = file_reg + str(a) + 'start.csv'
    uvc_pred, cart_pred = loc_class.read_predictions(file_phi_a,file_reg_a)
    # Compute localisation errors if ground truths are provided
    # loc_class.compute_LE(self,file_ground,cart_pred)
    # Print out predicted points (in cartesian and uvc space)
    out_cart = out_folder +  str(a) + 'start_cartesian_predictions.pts'
    out_uvc = out_folder +  str(a) + 'start_uvc_predictions.dat'
    print('Saving predictions in .pts format (cartesian): {} ...'.format(out_cart))
    with open(out_cart,'w') as f:
        f.write('{}'.format(cart_pred.shape[0]))
        [f.write('%.3f %.3f %.3f\n'%(i[0],i[1],i[2])) for i in cart_pred]
    print('Saving predictions in UVC format (.dat): {} ...'.format(out_uvc))
    np.savetxt(out_uvc,uvc_pred,delimiter=' ')
