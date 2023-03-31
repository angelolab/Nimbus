import warnings
from collections.abc import Iterable
import numpy as np
from typing import Any
import os
import pathlib


def create_invalid_data_str(invalid_data):
    """Creates a easy to read string for ValueError statements.
    Args:
        invalid_data (list[str]): A list of strings containing the invalid / missing data
    Returns:
        str: Returns a formatted string for more detailed ValueError outputs.
    """
    # Holder for the error string
    err_str_data = ""

    # Adding up to 10 invalid values to the err_str_data.
    for idx, data in enumerate(invalid_data[:10], start=1):
        err_msg = "{idx:{fill}{align}{width}} {message}\n".format(
            idx=idx,
            message=data,
            fill=" ",
            align="<",
            width=12,
        )
        err_str_data += err_msg

    return err_str_data


def make_iterable(a: Any, ignore_str: bool = True):
    """ Convert noniterable type to singleton in list
    Args:
        a (T | Iterable[T]):
            value or iterable of type T
        ignore_str (bool):
            whether to ignore the iterability of the str type
    Returns:
        List[T]:
            a as singleton in list, or a if a was already iterable.
    """
    return a if isinstance(a, Iterable) and not ((isinstance(a, str) and ignore_str) or
                                                 isinstance(a, type)) else [a]


def verify_in_list(warn=False, **kwargs):
    """Verify at least whether the values in the first list exist in the second
    Args:
        warn (bool):
            Whether to issue warning instead of error, defaults to False
        **kwargs (list, list):
            Two lists, but will work for single elements as well.
            The first list specified will be tested to see
            if all its elements are contained in the second.
    Raises:
        ValueError:
            if not all values in the first list are found in the second
        Warning:
            if not all values are found and warn is True
    """

    if len(kwargs) != 2:
        raise ValueError("You must provide 2 arguments to verify_in_list")

    test_list, good_values = kwargs.values()
    test_list = list(make_iterable(test_list))
    good_values = list(make_iterable(good_values))

    for v in [test_list, good_values]:
        if len(v) == 0:
            raise ValueError("List arguments cannot be empty")

    if not np.isin(test_list, good_values).all():
        test_list_name, good_values_name = kwargs.keys()
        test_list_name = test_list_name.replace("_", " ")
        good_values_name = good_values_name.replace("_", " ")

        # Calculate the difference between the `test_list` and the `good_values`
        difference = [str(val) for val in test_list if val not in good_values]

        # Only printing up to the first 10 invalid values.
        err_str = ("Not all values given in list {0:^} were found in list {1:^}.\n "
                   "Displaying {2} of {3} invalid value(s) for list {4:^}\n").format(
            test_list_name, good_values_name,
            min(len(difference), 10), len(difference), test_list_name
        )

        err_str += create_invalid_data_str(difference)

        if warn:
            warnings.warn(err_str)
        else:
            raise ValueError(err_str)


def validate_paths(paths, data_prefix=True):
    """Verifys that paths exist and don't leave Docker's scope
    Args:
        paths (str or list):
            paths to verify.
        data_prefix (bool):
            if True, checks that directory starts with /data, necessary when inside the docker
    Raises:
        ValueError:
            Raised if any directory is out of scope or non-existent
    """

    # if given a single path, convert to list
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            if str(path).startswith('../data') or not data_prefix:
                for parent in reversed(pathlib.Path(path).parents):
                    if not os.path.exists(parent):
                        raise ValueError(
                            f'A bad path, {path}, was provided.\n'
                            f'The folder, {parent.name}, could not be found...')
                raise ValueError(
                    f'The file/path, {pathlib.Path(path).name}, could not be found...')
            else:
                raise ValueError(
                    f'The path, {path}, is not prefixed with \'../data\'.\n'
                    f'Be sure to add all images/files/data to the \'data\' folder, '
                    f'and to reference as \'../data/path_to_data/myfile.tif\'')


def list_folders(dir_name, substrs=None, exact_match=False, ignore_hidden=True):
    """ List all folders in a directory containing at least one given substring
    Args:
        dir_name (str):
            Parent directory for folders of interest
        substrs (str or list):
            Substring matching criteria, defaults to None (all folders)
        exact_match (bool):
            If True, will match exact folder names (so 'C' will match only 'C/').
            If False, will match substr pattern in folder (so 'C' will match 'C/' & 'C_DIREC/').
        ignore_hidden (bool):
            If True, will ignore hidden directories. If False, will allow hidden directories to
            be matched against the search substring.
    Returns:
        list:
            List of folders containing at least one of the substrings
    """
    files = os.listdir(dir_name)
    folders = [file for file in files if os.path.isdir(os.path.join(dir_name, file))]

    # Filter out hidden directories
    if ignore_hidden:
        folders = [folder for folder in folders if not folder.startswith('.')]

    # default to return all files
    if substrs is None:
        return folders

    # handle case where substrs is a single string (not wrapped in list)
    if type(substrs) is not list:
        substrs = [substrs]

    # Exact match case
    if exact_match:
        matches = [folder
                   for folder in folders
                   if any([
                       substr == os.path.splitext(folder)[0]
                       for substr in substrs
                   ])]
    else:
        matches = [folder
                   for folder in folders
                   if any([
                       substr in folder
                       for substr in substrs
                   ])]

    return matches
