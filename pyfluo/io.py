import time
import os
import pickle
from os.path import isfile, isdir, join, basename, splitext


def save(file_name, **items):
    """Save any number of objects in the current workspace into a single file.
    
    Args:
            file_name (str): name of saved file containing objects
            **items (keyword-object pairs or unpacked dictionary): pairs or dictionary in which the key is the object name and the value is the object

    Example:
        >>> object_a = 'hello'
        >>> object_b = np.array([1,2,3])
        >>> save('my_saved_objects', a=object_a, b=object_b)
    """     
    f = open(file_name, 'wb')
    pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)
    f.close()
def load(file_name):
        """Load previously saved object/s.
        
        Args:
                file_name (str): file to load
                
        Notes:
                When used from a script, this function can be used to load the saved objects directly into the workspace, as follows:
                globals().update(load(path))
                (But note that this will overwrite current variables in the workspace if their names are the same.)
                
        Returns:
                Dictionary of loaded objects, as {name: obj ...}
        """
        return pickle.load(open(file_name, 'rb'))
        
        
