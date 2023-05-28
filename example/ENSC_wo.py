import sys
sys.path.append('.')
from tool.ClusteringEvaluator import cluster_accuracy
from method.ENSC_wo import ENSC_wo
import numpy as np
from tool.show_class_map import show_class_map



if __name__ == '__main__':
    from tool.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale, normalize
    from sklearn.decomposition import PCA
    import time

    root = '../HSI_Datasets/'
    #im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    #im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    im_, gt_ = 'PaviaU', 'PaviaU_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print('\nDataset: ', img_path)

    PATCH_SIZE = 9  # # 9 default, normally the bigger the better
    nb_comps = 4  # # num of PCs, 4 default, it can be moderately increased
    # load img and gt
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)

    # # take a smaller sub-scene for computational efficiency
    if im_ == 'SalinasA_corrected':
        REG_Coef_, NEIGHBORING_, RO_ = 1e1, 30, 0.8
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e2, 30, 0.8, 0.2
    if im_ == 'Indian_pines_corrected':
        img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        PATCH_SIZE = 13
        REG_Coef_, NEIGHBORING_, RO_ = 1e2, 30, 0.4
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e3, 30, 0.8, 10
    if im_ == 'PaviaU':
        img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        REG_Coef_, NEIGHBORING_, RO_ = 1e3, 20, 0.6
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 6 * 1e4, 30, 0.8, 100

    n_row, n_column, n_band = img.shape
    x_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    print('original img shape: ', x_img.shape)
    # # reduce spectral bands using PCA
    pca = PCA(n_components=nb_comps)
    img = minmax_scale(pca.fit_transform(img.reshape(n_row * n_column, n_band))).reshape(n_row, n_column, nb_comps)
    x_patches, y_ ,nonzero_index= p.get_HSI_patches_rw(img, gt, (PATCH_SIZE, PATCH_SIZE))

    print('reduced img shape: ', img.shape)
    print('x_patch tensor shape: ', x_patches.shape)
    n_samples, n_width, n_height, n_band = x_patches.shape
    x_patches_2d = np.reshape(x_patches, (n_samples, -1))
    y = p.standardize_label(y_)

    x_patches_2d = normalize(x_patches_2d)
    print('final sample shape: %s, labels: %s' % (x_patches_2d.shape, np.unique(y)))
    N_CLASSES = np.unique(y).shape[0]  # Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

    # ========================
    # performing  ENSC
    # ========================
    time_start = time.perf_counter()
    edsc = ENSC_wo(n_clusters=N_CLASSES, regu_coef=REG_Coef_, n_neighbors=NEIGHBORING_, ro=RO_, save_affinity=False)
    y_pre_edsc = edsc.fit(x_patches_2d)
    run_time = round(time.perf_counter() - time_start, 3)
    acc = cluster_accuracy(y, y_pre_edsc)
    show_class_map(y_pre_edsc,nonzero_index,gt)

    print('=================================\n'
          '\t\tENSC(w/o enhance) RESULTS\n'
          '=================================')
    print('%10s %10s %10s' % ('OA', 'Kappa', 'NMI',))
    print('%10.4f %10.4f %10.4f' % (acc[0], acc[1], acc[2]))
    print('class accuracy:', acc[3])
    print('running time', run_time)



