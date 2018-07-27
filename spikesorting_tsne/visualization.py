
import matplotlib.pyplot as plt
from . import constants as ct


def plot_tsne_of_spikes(spike_info, cm=None, subtitle=None, label_name='Template', label_array=None,
              legent_on=True, axes=None, max_screen=False, hide_ticklabels=False):

    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.tight_layout()
    else:
        ax = axes

    if hide_ticklabels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if max_screen:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    if subtitle is None and ax is None:
        fig.suptitle('T-SNE')
    elif ax is None:
        fig.suptitle(subtitle)

    if cm is None:
        cm = plt.cm.Dark2

    alpha = 1
    markers = ['.', 'o', 's', '*', 'D', 'v', '>', '<']

    templates_used = spike_info[ct.TEMPLATE_AFTER_SORTING].unique()
    number_of_templates = len(templates_used)
    color_indices = plt.Normalize(0, templates_used.max())

    labeled_scatters = []

    for template in templates_used:
        tsne_x = spike_info[spike_info[ct.TEMPLATE_AFTER_SORTING] == template][ct.TSNE_X]
        tsne_y = spike_info[spike_info[ct.TEMPLATE_AFTER_SORTING] == template][ct.TSNE_Y]
        full_template = \
            spike_info[spike_info[ct.TEMPLATE_AFTER_SORTING] == template][ct.TEMPLATE_WITH_ALL_SPIKES_PRESENT].iloc[0]
        if full_template:
            size = 10
        else:
            size = 40
        type_of_template = spike_info[spike_info[ct.TEMPLATE_AFTER_SORTING] == template][ct.TYPE_AFTER_SORTING].iloc[0]
        for t in range(len(ct.types)):
            if ct.types[t] == type_of_template:
                marker = markers[t]
        labeled_scatters.append(ax.scatter(tsne_x, tsne_y, s=size, color=cm(color_indices(template)),
                                           alpha=alpha, marker=marker))

    if legent_on:
        ncol = int(number_of_templates / 40)
        box = ax.get_position()
        ax.set_position([0.03, 0.03, box.width * (1 - 0.04 * ncol), 0.93])
        if label_array is None:
            label_array = np.array(range(number_of_templates))
        if label_array.dtype == int:
            threshold_legend = np.char.mod('{} %i'.format(label_name), label_array)
        if label_array.dtype == float:
            threshold_legend = np.char.mod('{} %f'.format(label_name), label_array)
        else:
            threshold_legend = label_array
        plt.legend(labeled_scatters, threshold_legend, scatterpoints=1, ncol=ncol, loc='center left',
                   bbox_to_anchor=(1.0, 0.5))
    else:
        plt.tight_layout(rect=[0, 0, 1, 1])

    if axes is None:
        return fig, ax
    else:
        pass








def make_video_of_tsne_iterations(spike_info, iterations, video_dir, data_file_name='interim_{:0>6}.dat',
                                  video_file_name='tsne_video.mp4', figsize=(15, 15), dpi=200, fps=30,
                                  movie_metadata=None, cm=None, subtitle=None,label_name='Label',
                                  legent_on=False, max_screen=False):
    iters = np.arange(iterations)
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = None
    if movie_metadata:
        metadata = movie_metadata
    writer = FFMpegWriter(fps=fps, bitrate=-1, metadata=metadata)
    if cm is None:
        cm = plt.cm.Dark2
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    with writer.saving(fig, join(video_dir, video_file_name), dpi):
        for it in iters:
            ax.cla()
            tsne = io.load_tsne_result(video_dir, data_file_name.format(it))
            tsne = np.transpose(tsne)
            spike_info[ct.TSNE_X] = tsne[0, :]
            spike_info[ct.TSNE_Y] = tsne[1, :]
            plot_tsne_of_spikes(spike_info, cm=cm, subtitle=subtitle, label_name=label_name, label_array=cm_remapping,
                                axes=ax, legent_on=legent_on, max_screen=max_screen, hide_ticklabels=True)
            min_x = np.min(tsne[0, :])
            max_x = np.max(tsne[0, :])
            min_y = np.min(tsne[1, :])
            max_y = np.max(tsne[1, :])
            range_x = np.max(np.abs([min_x, max_x]))
            range_y = np.max(np.abs([min_y, max_y]))

            plt.ylim([-range_y, range_y])
            plt.xlim([-range_x, range_x])
            writer.grab_frame()
            if it%100 == 0:
                print('Done '+str(it) + ' frames')






