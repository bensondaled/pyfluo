import pickle

def save(file_name, **items):
    """Save any number of objects in the current workspace into a single file
    
    Parameters
    ----------
    file_name (str): name of file to save
    items (keyword-object pairs or unpacked dictionary): pairs or dictionary in which the key is the object name and the value is the object

    Example:
        >>> object_a = 'hello'
        >>> object_b = np.array([1,2,3])
        >>> save('my_saved_objects', a=object_a, b=object_b)
    """     

    with open(file_name, 'wb') as f:
        pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)
def load(file_name):
        """Load previously saved object/s
        
        Parameters
        ----------
        file_name (str): file to load
        
        Returns
        -------
        Dictionary of loaded objects, as {name: obj ...}
         
        Note: this function can be used to load the saved objects directly into the workspace, as follows (variables with matching names will be overwritten):
        >>> globals().update(load(path))
        """
        with open(file_name, 'rb') as f:
            return pickle.load(f)
        
