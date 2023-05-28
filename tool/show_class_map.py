import copy
import matplotlib.pyplot as plt
def show_class_map(y_pre, y_indx, gt, show=True, save=False):

    gt_pre = copy.deepcopy(gt)
    gt_pre[y_indx] = y_pre
    fig, ax = plt.subplots()
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # cmap = ListedColormap(np.array(spy_colors) / 255.)
    # cmap = (np.array(self.class_colors) / 255.)
    ax.imshow(gt_pre, cmap='nipy_spectral')  # spectral
    plt.axis('off')
    plt.tight_layout()
    if save is not False:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    if show:
        plt.show()
def show_gt(gt, show=True, save=False):

    fig, ax = plt.subplots()
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # cmap = ListedColormap(np.array(spy_colors) / 255.)
    # cmap = (np.array(self.class_colors) / 255.)
    ax.imshow(gt, cmap='nipy_spectral')  # spectral
    plt.axis('off')
    plt.tight_layout()
    if save is not False:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    if show:
        plt.show()