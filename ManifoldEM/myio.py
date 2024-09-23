import h5py, pickle
from typing import Dict, Any


def fin1(filename: str) -> Dict[str, Any]:
    """
    Loads the data from the given file using Python's pickle deserialization.

    Parameters
    ----------
    filename : str
        The name of the file where the data will be loaded from.

    Returns
    -------
    dict
        A dictionary containing the data loaded from the file.
    """
    if filename.endswith(".h5"):
        with h5py.File(filename, "r") as f:
            return {key: f[key][()] for key in f.keys()}
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def fout1(filename: str, **kwargs) -> None:
    """
    Saves the given keyword arguments to a file using Python's pickle serialization.

    Parameters
    ----------
    filename : str
        The name of the file where the data will be saved.
    **kwargs
        Arbitrary keyword arguments to be saved.
    """

    if filename.endswith(".h5"):
        with h5py.File(filename, "w") as f:
            for key, value in kwargs.items():
                f.create_dataset(key, data=value)
    else:
        with open(filename, "wb") as f:
            pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
