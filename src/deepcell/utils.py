
# Copyright 2016-2023 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow.python.client import device_lib


def count_gpus():
    """Get the number of available GPUs.

    Returns:
        int: count of GPUs as integer
    """
    devices = device_lib.list_local_devices()
    gpus = [d for d in devices if d.name.lower().startswith('/device:gpu')]
    return len(gpus)


def get_sorted_keys(dict_to_sort):
    """Gets the keys from a dict and sorts them in ascending order.
    Assumes keys are of the form ``Ni``, where ``N`` is a letter and ``i``
    is an integer.

    Args:
        dict_to_sort (dict): dict whose keys need sorting

    Returns:
        list: list of sorted keys from ``dict_to_sort``
    """
    sorted_keys = list(dict_to_sort.keys())
    sorted_keys.sort(key=lambda x: int(x[1:]))
    return sorted_keys