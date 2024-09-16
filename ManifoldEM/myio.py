import pickle
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
    with open(filename, 'rb') as f:
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

    with open(filename, 'wb') as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
