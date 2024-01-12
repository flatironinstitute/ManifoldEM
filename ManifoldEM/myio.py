import pickle
from typing import Dict, Any


def fin1(filename: str) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def fout1(filename: str, **kwargs) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

