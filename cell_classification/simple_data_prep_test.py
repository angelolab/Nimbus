import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from simple_data_prep import SimpleTFRecords
from segmentation_data_prep_test import prepare_test_data_folders, prepare_cell_type_table
import json
import tempfile


def test_get_marker_activity():
    with tempfile.TemporaryDirectory() as temp_dir:
        norm_dict = {"CD11c": 1.0, "CD4": 1.0, "CD56": 1.0, "CD57": 1.0}
        with open(os.path.join(temp_dir, "norm_dict.json"), "w") as f:
            json.dump(norm_dict, f)
        data_folders = prepare_test_data_folders(
            5, temp_dir, list(norm_dict.keys()) + ["XYZ"], random=True,
            scale=[0.5, 1.0, 1.5, 2.0, 5.0]
        )
        cell_table_path = os.path.join(temp_dir, "cell_table.csv")
        cell_table = pd.DataFrame(
           {
                "SampleID": ["fov_0"] * 15 + ["fov_1"] * 15 + ["fov_2"] * 15 + ["fov_3"] * 15 +
                ["fov_4"] * 15 + ["fov_5"] * 15,
                "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 6,
                "CD4_gt": [1, 1, 0, 0, 2] * 3 * 3 +
                [0, 0, 1, 1, 2] * 3 * 3,
            }
        )

        cell_table.to_csv(cell_table_path, index=False)
        data_prep = SimpleTFRecords(
            data_dir=temp_dir,
            tf_record_path=temp_dir,
            cell_table_path=cell_table_path,
            normalization_dict_path=None,
            selected_markers=["CD4"],
            imaging_platform="test",
            dataset="test",
            tile_size=[256, 256],
            stride=[256, 256],
            nuclei_channels=["CD56"],
            membrane_channels=["CD57"],
        )
        data_prep.load_and_check_input()
        marker = "CD4"
        sample_name = "fov_1"
        fov_1_subset = cell_table[cell_table.SampleID == sample_name]
        data_prep.sample_subset = fov_1_subset
        marker_activity, _ = data_prep.get_marker_activity(sample_name, marker)

        # check if the we get marker_acitivity for all labels in the fov_1 subset
        assert np.array_equal(marker_activity.labels, fov_1_subset.labels)

        # check if the df has the right marker activity values for a given cell
        assert np.array_equal(marker_activity.activity.values, fov_1_subset.CD4_gt.values)
