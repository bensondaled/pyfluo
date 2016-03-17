import re, os, shutil
from .config import *

"""
This module is inteded to help organize imaging data and its associated data.
A "Group" is simply a directory that stores:
        -movie files (tifs)
        -motion correction data (hdf5)
"""

class Group():
    """
    Specifies a group of movies and their associated rois, series, motion corrections, etc.
    Can technically comprise any combination of objects, but currently intended to store info from 1 FOV over n imaging epochs (whether across days or not)
    """
    def __init__(self, name, path=PF_data_path):
        self.name = name
        self.grp_path = os.path.join(path, name)

        if not os.path.exists(self.grp_path):
            raise Exception('Group does not exist. You can use Group.from_raw to make the group.')

    @classmethod
    def from_raw(self, name, in_path='.', out_path=PF_data_path, regex=r'.*', move_files=True):
        """Creates and returns a new group based on the input data
        """

        # create group
        grp_path = os.path.join(out_path, name)
        if os.path.exists(grp_path):
            raise Exception('Group \'{}\' already exists.'.format(grp_path))
        os.mkdir(grp_path)
        
        # load files in specified directory
        filenames = [os.path.join(in_path, p) for p in os.listdir(in_path)]

        # filter files
        filenames = [fn for fn in filenames if re.search(regex, fn)]

        # move/copy files
        if move_files:
            mv_func = shutil.move
        else:
            mv_func = shutil.copy2
        for fn in filenames:
            mv_func(fn, grp_path)
        
        return Group(name, path=out_path)
