import sys
sys.path.append('.')
from tool.show_class_map import show_gt

if __name__ == '__main__':
    from tool.Preprocessing import Processor


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


    show_gt(gt)



