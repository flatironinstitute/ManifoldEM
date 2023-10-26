from importlib.metadata import distribution, PackageNotFoundError
try:
    __version__ = distribution(__name__).version
except PackageNotFoundError:
    # package is not installed
    pass
