import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import colorcet as cc


label_dict = {
    'null': 0,
    'jogging': 1,
    'jogging (rotating arms)': 2,
    'jogging (skipping)': 3,
    'jogging (sidesteps)': 4,
    'jogging (butt-kicks)': 5,
    'stretching (triceps)': 6,
    'stretching (lunging)': 7,
    'stretching (shoulders)': 8,
    'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10,
    'push-ups': 11,
    'push-ups (complex)': 12,
    'sit-ups': 13,
    'sit-ups (complex)': 14,
    'burpees': 15,
    'lunges': 16,
    'lunges (complex)': 17,
    'bench-dips': 18
}

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def compare_timelines(gt, timeline_1, timeline_2, timeline_3, timeline_4, timeline_5):
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter

    n_classes = len(np.unique(gt))

    # plot 1:
    fig, (gt_ax, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, sharex=True, figsize=(9, 6), layout="constrained")

    colors1 = sns.color_palette(palette=cc.glasbey_warm, n_colors=n_classes).as_hex()
    colors1[0] = '#F8F8F8'
    cmap1 = LinearSegmentedColormap.from_list(name='My Colors1', colors=colors1, N=len(colors1))

    gt_ax.set_yticks([])
    gt_ax.pcolor([gt], cmap=cmap1, vmin=0, vmax=n_classes)

    ax1.set_yticks([])
    ax1.pcolor([timeline_1], cmap=cmap1, vmin=0, vmax=n_classes)

    ax2.set_yticks([])
    ax2.pcolor([timeline_2], cmap=cmap1, vmin=0, vmax=n_classes)

    ax3.set_yticks([])
    ax3.pcolor([timeline_3], cmap=cmap1, vmin=0, vmax=n_classes)
   
    ax4.set_yticks([])
    ax4.pcolor([timeline_4], cmap=cmap1, vmin=0, vmax=n_classes)
    
    ax5.set_yticks([])
    ax5.pcolor([timeline_5], cmap=cmap1, vmin=0, vmax=n_classes)
    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    print(colors1)

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.xticks([])
    plt.savefig('test.png')
    plt.close()


def get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt

    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

time_1 = np.load('predictions/aandd_751_inertial.npy')
time_2 = np.load('predictions/tridet_0.2_camera.npy')
time_3 = np.load('predictions/tridet_0.2_combined.npy')
time_4 = np.load('predictions/oracle.npy')
time_5 = np.load('predictions/oracle_combined.npy')

sbjs_s = ['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5', 'sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11', 'sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']
all_s_data = np.empty((0, 12 + 2))
for sbj in sbjs_s:
    t_data = pd.read_csv(os.path.join('data/wear/raw/inertial', sbj + '.csv'), index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
    all_s_data = np.append(all_s_data, t_data, axis=0)

gt_viz = all_s_data[(all_s_data[:, 0] == 12)][:, -1]
time_1_viz = time_1[(all_s_data[:, 0] == 12)]
time_2_viz = time_2[(all_s_data[:, 0] == 12)]
time_3_viz = time_3[(all_s_data[:, 0] == 12)]
time_4_viz = time_4[(all_s_data[:, 0] == 12)]
time_5_viz = time_5[(all_s_data[:, 0] == 12)]

compare_timelines(gt_viz, time_1_viz, time_2_viz, time_3_viz, time_4_viz, time_5_viz)
