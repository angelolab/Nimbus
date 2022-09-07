from ast import arg
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from segmentation_data_prep import parse_dict, feature_description
import tensorflow as tf
from skimage.segmentation import find_boundaries
import numpy as np
import argparse
from tqdm import tqdm


def segmentation_to_boundaries(labels):
    """Convert a segmentation to a binary mask of the boundaries

    Args:
        segmentation (np.ndarray):
            A 2D array of integers representing a segmentation

    Returns:
        np.ndarray:
            A 2D array of booleans representing the boundaries of the segmentation
    """
    boundaries = find_boundaries(labels, mode="inner")
    boundaries = labels * boundaries.astype(labels.dtype)
    return boundaries


def plot_overlay(example, save_dir=None, save_file=None, dpi=160):
    """Plot the marker image and the marker activity segmentation overlayed

    Args:
        example (dict):
            Dictionary with keys "mplex_img", "marker_activity_mask"
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """
    marker_boundaries = segmentation_to_boundaries(example["marker_activity_mask"].numpy())
    instance_mask = example["instance_mask"].numpy()
    instance_mask[example["marker_activity_mask"].numpy() > 0] = 0
    other_boundaries = segmentation_to_boundaries(instance_mask) > 0
    img = np.repeat(example["mplex_img"], 3, axis=-1)
    img[np.squeeze(other_boundaries)] = 0.2
    colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    for i, val in enumerate(np.unique(marker_boundaries)):
        if val == 0:
            continue
        img[np.squeeze(marker_boundaries) == val] = colors[i]
    pos = mpatches.Patch(color=(0, 1, 0), label="Positive")
    neg = mpatches.Patch(color=(0, 0, 0), label="Negative")
    und = mpatches.Patch(color=(0, 0, 1), label="Undetermined")
    oth = mpatches.Patch(color=(0.4, 0.4, 0.4), label="Others")
    ax = plt.subplot(111)
    ax.imshow(img.clip(0, 1), interpolation="nearest")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    plt.legend(
        handles=[pos, neg, und, oth],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        ncol=3,
    )
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


def plot_together(example, save_dir=None, save_file=None, dpi=160):
    colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("BRGB", colors, N=4)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(example["mplex_img"].numpy().clip(0, 1), interpolation="nearest", cmap="gray")
    ax[1].imshow(example["binary_mask"].numpy(), interpolation="nearest", cmap="gray")
    ax[2].imshow(
        example["marker_activity_mask"].numpy(), interpolation="nearest", cmap=cmap, vmin=0, vmax=3
    )
    ax[0].title.set_text("mplex img")
    ax[1].title.set_text("binary_mask")
    ax[2].title.set_text("marker_activity_mask")
    pos = mpatches.Patch(color=colors[1], label="Positive")
    neg = mpatches.Patch(color=colors[0], label="Negative")
    und = mpatches.Patch(color=colors[3], label="Undetermined")
    for axx in ax:
        box = axx.get_position()
        axx.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax[1].legend(
        handles=[pos, neg, und],
        loc="upper center",
        bbox_to_anchor=(-0.1, -0.2),
        fancybox=True,
        ncol=3,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record_path",
        type=str,
        default="C:/Users/lorenz/Downloads/Lorenz_example_data/TNBC.tfrecord",
    )
    parser.add_argument(
        "--save_dir", type=str, default="C:/Users/lorenz/Downloads/Lorenz_example_data/plots"
    )
    parser.add_argument("--dpi", type=float, default=160)
    parser.add_argument("--plot_overlay", default=True)
    parser.add_argument("--shuffle", default=True)
    args = parser.parse_args()
    path = args.record_path
    save_dir = args.save_dir
    dpi = args.dpi
    train_ds = tf.data.TFRecordDataset(path)
    if args.shuffle:
        train_ds = train_ds.shuffle(300)
    for i, record in tqdm(enumerate(train_ds)):
        example_encoded = tf.io.parse_single_example(record, feature_description)
        example = parse_dict(example_encoded)
        plot_overlay(
            example,
            dpi=dpi,
            save_dir=save_dir,
            save_file=f"{example['folder_name']}_{example['marker']}_overlay_{i}.png",
        )
        plot_together(
            example,
            dpi=dpi,
            save_dir=save_dir,
            save_file=f"{example['folder_name']}_{example['marker']}_together_{i}.png",
        )
