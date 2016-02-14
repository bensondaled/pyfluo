import numpy as np
import ts_base, movies, traces
Movie = movies.Movie
Trace = traces.Trace
TSBase = ts_base.TSBase

def save(file_name, **items):
    """Save any number of objects in the current workspace into a single file
    
    Parameters
    ----------
    file_name : str
        name of file to save
    items : keyword-object pairs or unpacked dictionary 
        pairs or dictionary in which the key is the object name and the value is the object

    Example:
        >>> object_a = 'hello'
        >>> object_b = np.array([1,2,3])
        >>> save('my_saved_objects', a=object_a, b=object_b)
    """     

    for item in items:
        obj = items[item]
        if isinstance(obj, TSBase):
            del items[item]
            items[item] = obj.decompose()
        elif isinstance(obj, np.ndarray):
            pass
        else:
            raise Exception('Package does not explicitly support saving of this data type. Try saving using np.save or pickle.')

    if not file_name.endswith('.pyfluo'):
        file_name += '.pyfluo'
    np.savez(file_name, **items)

def load(file_name):
        """Load previously saved object/s
        
        Parameters
        ----------
        file_name : str
            file to load
        
        Returns
        -------
        Dictionary of loaded objects, as {name: obj ...}
         
        Note: this function can be used to load the saved objects directly into the workspace, as follows (variables with matching names will be overwritten):
        >>> globals().update(load(path))
        """
        res = {}
        with np.load(file_name) as f:
            for k in f:
                obj = f[k]
                if obj.dtype.names and 'class' in obj.dtype.names:
                    obj = obj[0]
                    try:
                        dest_class = eval(obj['class'])
                    except:
                        raise Exception('Class \'%s\' of saved object is not recognized.'%str(obj['class']))

                    args_dict = {n:obj[n] for n in obj.dtype.names if n!='class'}
                        
                    obj = dest_class(**args_dict)
                res[k] = obj
        return res
        
