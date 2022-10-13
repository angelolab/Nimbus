import numpy as np
import pandas as pd


def process_to_cells(instance_mask, prediction):
    """Process predictions from pixel level to cell level
    Args:
        instance_mask (np.array):
            2D array of instance mask
        prediction (np.array):
            2D array of pixel level prediction
    Returns:
        np.array:
            2D array of cell level averaged prediction
        pd.DataFrame:
            DataFrame of cell level predictions
    """

    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels != 0]
    mean_per_cell_mask = np.zeros_like(instance_mask, dtype=np.float32)
    df = pd.DataFrame(columns=['labels', 'pred_activity'])
    i = 0
    for unique_label in unique_labels:
        mask = instance_mask == unique_label
        mean_pred = prediction[mask].mean()
        mean_per_cell_mask[mask] = mean_pred
        df = pd.concat(
            [df, pd.DataFrame(
                {'labels': [unique_label], 'pred_activity': [mean_pred]}, index=[i]
            )],
        )
        i += 1
    return mean_per_cell_mask, df


def merge_activity_df(gt_df, pred_df):
    """Merge ground truth and prediction dataframes over labels
    Args:
        gt_df (pd.DataFrame):
            DataFrame of ground truth
        pred_df (pd.DataFrame):
            DataFrame of prediction
    Returns:
        pd.DataFrame:
            DataFrame of merged ground truth and prediction
    """
    pred_df.labels = pred_df.labels.astype(int)
    gt_df.labels = gt_df.labels.astype(int)
    df = gt_df.merge(pred_df, on='labels', how='left')
    return df
