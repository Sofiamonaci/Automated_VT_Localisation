"""
Compute localisation errors between AI predicted exit sites and known ground truths (from either simulations or ablation targets)

This function provides methods to compute localisation errors, plot/visualise predicted points

"""
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass
class Localisation:
    """
    Class to process AI prediction results (for both PHI classification and Z_RHO regression),
    compute localisation errors against known sites and plot/visualise such results.

    Args:
        uvc_path (str):     path to uvc coordinates of mesh of interest (COMBINED_Z_RHO_PHI_V.dat)
        mesh_path (str):    path to mesh of interest (.pts format)
        aha_path (str):     path to 17 aha segments (.dat format)
        classes_path (str): path to 17 PHI classes (.dat format) -> 1 to 17

    """

    uvc_path: str
    mesh_path: str
    aha_path: str
    classes_path: str


    def __post_init__(self):

        print('Reading files: {}\n\t{}\n\t{}\n\t{}\n'.format(self.uvc_path,self.mesh_path,self.aha_path,self.classes_path))
        self.uvc = pd.read_csv(self.uvc_path, header=None).values
        self.mesh = pd.read_csv(self.mesh_path, skiprows=1, header=None, delimiter=' ').values
        self.aha = pd.read_csv(self.aha_path, header=None).values
        self.classes = pd.read_csv(self.classes_path, header=None).values

    def convert_classes_to_phi(self, pred):

        """
        convert PHI classes into actual PHI
        :param pred:    (np.arrray) predicted classes (either categorical or 1D)
        :return:        out_pred (phi)
        """

        print('Convert PHI classes into PHI continous values...\n\n')
        # If classes is nd.array, convert categorical to single array
        if pred.ndim>1:
            pred = np.where(pred)[0] + 1 # because classes are saved from 1 to 17

        out_pred = pred*0
        for i in range(min(pred),max(pred)):
            out_pred[pred==i] = np.mean(self.uvc[self.classes==i,2])

        return out_pred

    def convert_uvc_to_pts(self,
                           query,
                           outfile=''):

        """
        Convert uvc to cartesian.

        :param uvc_prediction:  (str) uvc prediction in the format ([z,rho,phi,v])
        :param outfile:         (str) output file (default = empty, not outputting)

        :return:                pts_pred
        """

        # Unpack v for ease of coding
        v = self.uvc[:,-1]

        # scaling for uvc to pts (please refer to Bayer et al. paper on UVC)
        s = [3, 0.33, 1]
        # distance between uvc x and uvc y based on s scaling
        d = lambda x,y: (s[0]**2*(y[0]-x[0])**2 + s[1]**2*(y[1]-x[1])**2 + s[2]**2*(y[2]-x[2])**2)**0.5

        for i in range(query.shape[0]):

            pts_i = self.mesh[v==query[:,-1],:]
            uvc_i = self.uvc[v==query[:,-1],:-1]

            # Find index of closest point to mesh
            ind_query = d(uvc_i,query[:,:-1]).argmin()
            pts_pred = pts_i[ind_query,:]


        # Printing cartesian prediction in .csv format if outfile is not empty
        if outfile:
            print('Printing out converted predictions in {}...'.format(outfile))
            np.savetxt(outfile,pts_pred)

        return pts_pred



# For now, the network only predicts LV points, so in the function v values are assigned to -1 automatically
    def read_predictions(self,
                        file_phi,
                        file_reg,
                        outfile):

        """
        Read csv files generated when testing AI architecture, compute UVC predicctions and convert them
        to cartesian space

        :param file_phi:    (str) PHI classes prediction .csv
        :param file_reg:    (str) Z_RHO regression values prediction .csv
        :param outfile:     (str) output file where to save cartesian prediction
        :return:            uvc_prediction, cart_prediction
        """


        print('\n\nCOMPUTING LOCALISATION ERRORS FOR PREDICTED AI POINTS\n\n')
        print('Reading {}...'.format(file_phi))
        pred = pd.read_csv(file_phi, header=None).values
        phi = self.convert_classes_to_phi(pred)
        print('Reading {}...'.format(file_reg))
        reg = pd.read_csv(file_reg, header=None).values

        # uvc_predictions
        v = np.zeros(phi.shape) - 1
        uvc_prediction = [reg,phi,v]
        # cartesian_predictions
        cart_prediction = self.convert_uvc_to_pts(uvc_prediction,outfile)

        return uvc_prediction, cart_prediction

    def compute_LE(self,
                   file_ground,
                   pred,
                   scale = 1000):

        """
        Compute localisation error with ground truth(s) if file is not empty

        :param file_ground:     (str) filepath to ground truth(s) in cartesian --> accepted formats: .pts or .csv
        :param pred:            (np.ndarray) predicted points
        :param scale:           (int) units of mesh in mm --> default: 1000mm as the mesh in microm

        :return:                return LE (array or int) and also plot histogram with mean LE and/or print mean LE
        """

        if file_ground:
            print('Ground truth(s) is not empty, computing LE ...\n\n')
            print('Reading {} ...\n'.format(file_ground))
            if '.csv' in file_ground:
                ground = pd.read_csv(file_ground, header=None).values
            elif '.pts' in file_ground:
                ground = pd.read_csv(file_ground, header=None, skiprows=1, delimiter=' ').values

            # Computing localisation error
            le = np.linalg.norm(ground-pred, axis=1)/scale

            # Compute mean and std localisation and plot histogram
            if len(le)>1:
                mean_le = np.mean(le)
                std_le = np.std(le)
                print('Plotting distribution of localisation errors ...')
                plt.hist(le)
                plt.xlabel('LE (mm)')
                plt.ylabel('Counts')
                plt.title('Mean LE: {} +- {} mm'.format(mean_le,std_le))
            else:
                print('LE: {} mm'.format(le))

        return le
