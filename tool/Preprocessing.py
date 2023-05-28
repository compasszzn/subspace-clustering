# coding: utf-8

from __future__ import print_function
import numpy as np
from tool.rolling_window import rolling_window as rw

class Processor:
    def __init__(self):
        pass

    def prepare_data(self, img_path, gt_path):
        if img_path[-3:] == 'mat':
            import scipy.io as sio
            img_mat = sio.loadmat(img_path)
            gt_mat = sio.loadmat(gt_path)
            img_keys = img_mat.keys()
            gt_keys = gt_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float64'), gt_mat.get(gt_key[0]).astype('int8')
        else:
            import spectral as spy
            img = spy.open_image(img_path).load()
            gt = spy.open_image(gt_path)
            a = spy.principal_components()
            a.transform()
            return img, gt.read_band(0)

    def get_HSI_patches_rw(self, x, gt, ksize, stride=(1, 1), padding='reflect', indix=True):
        """
        extract HSI spectral-spatial
        :param x: 3-D HSI (n_row, n_clm, n_band)
        :param gt: 2-D ground truth
        :param ksize: must be odd numbers, i.e. (3, 3) (7,7) ...
        :param stride:
        :param padding: padding mode: constant, reflect (default), etc.
        :return: (n_sample, ksize1, ksize2, n_band)
        """
        # # padding with boundary pixels
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (n_row, n_clm, n_band))
        y = np.reshape(gt, (n_row, n_clm))
        ksizes_ = (ksize[0], ksize[1])
        x_patches = rw(x, ksizes_, axes=(1, 0))  # divide data into 5x5 blocks
        y_patches = rw(y, ksizes_, axes=(1, 0))
        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        nonzero_index = y_patches[:, :, i_1, i_2].nonzero()
        x_patches_nonzero = x_patches[nonzero_index]
        y_patches_nonzero = (y_patches[:, :, i_1, i_2])[nonzero_index]
        x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 2, 3, 1])
        if indix is True:
            return x_patches_nonzero, y_patches_nonzero, nonzero_index
        return x_patches_nonzero, y_patches_nonzero


    def score(self, y_test, y_predicted):
        """
        calculate the accuracy and other criterion according to predicted results
        :param y_test:
        :param y_predicted:
        :return: ca, oa, aa, kappa
        """
        from sklearn.metrics import accuracy_score
        '''overall accuracy'''
        oa = accuracy_score(y_test, y_predicted)
        '''average accuracy for each classes'''
        n_classes = max([np.unique(y_test).__len__(), np.unique(y_predicted).__len__()])
        ca = []
        for c in np.unique(y_test):
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        aa = ca.mean()

        '''kappa'''
        kappa = self.kappa(y_test, y_predicted)
        return ca, oa, aa, kappa


    def kappa(self, y_test, y_predicted):
        from sklearn.metrics import cohen_kappa_score
        return round(cohen_kappa_score(y_test, y_predicted), 3)

    def color_legend(self, color_map, label):
        """

        :param color_map: 1-n color map in range 0-255
        :param label: label list
        :return:
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        size = len(label)
        patchs = []
        m = 255.  # float(color_map.max())
        color_map_ = (color_map / m)[1:]
        for i in range(0, size):
            patchs.append(mpatches.Patch(color=color_map_[i], label=label[i]))
        # plt.legend(handles=patchs)
        return patchs

    def standardize_label(self, y):
        """
        standardize the classes label into 0-k
        :param y: 
        :return: 
        """
        import copy
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
        return standardize_y
