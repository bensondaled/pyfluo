# External imports
import pandas as pd, numpy as np
import warnings, cv2

# Internal imports
from .config import *

class Series(pd.DataFrame):
    """Series object
    """

    _metadata = ['Ts']
    _metadata_defaults = [1.0]

    def __init__(self, data, *args, **kwargs):

        # Set custom fields
        for md,dmd in zip(self._metadata, self._metadata_defaults):
            setattr(self, md, kwargs.pop(md, dmd))

        # Init object
        super(Series, self).__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return Series
    
    @property
    def _constructor_sliced(self):
        return pd.Series

if __name__ == '__main__':
    pass

