import time
import os
import pickle
from os.path import isfile, isdir, join, basename, splitext

def pickle_save(obj, file_name):
	f = open(file_name, 'wb')
	pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	f.close()

def save(items=None, dir_name=None, glob=None):
	"""Save any number of objects in the current workspace, one file per object.
	
	Args:
		items (list/dict): the current workspace variables to be saved. If type list, it is converted to a dictionary where the variables names are the name attribute of the object.
		dir_name (str): name of directory in which to save object files, defaults to a time string.
		glob (dict): the built-in globals() dictionary in the scope from which save() is called.
		
	Notes:
		If the *items* argument is a dict, it should be in the form {'var_name': var}
		If it is a list, the name of each item is first cross-referenced with *glob*, and if not present, the object's *name* attribute is attempted.
	"""	
	if dir_name == None:
		 dir_name = time.strftime("saved-%Y%m%d_%H%M%S")

	if type(items) not in [dict, list]:
		items = [items]

	if type(items) == list:
		new_items = {}
		for item in items:
			if glob:
				new_entry = {g:glob[g] for g in glob if glob[g]==item and item not in new_items.items()}
				if len(new_entry):
					new_items.update(new_entry)
					continue
			try:
				new_entry = {item.name:item for item in items}
				new_items.update(new_entry)
			except AttributeError:
				print "Cannot determine a name for one of the given objects, so it will not be saved."
		
		items = new_items
			
	if dir_name not in os.listdir('.') or not isdir(dir_name):
		os.system('mkdir %s'%dir_name)
	os.chdir(dir_name)
	for name in items:
		try:
			pickle_save(items[name], name)
		except Exception:
			print "Could not save item: %s"%name
	os.chdir('..')
def load(path):
	"""Load previously saved object/s.
	
	Args:
		path (str): path to a file or directory.
		
	Notes:
		If path is to a file, loads the object saved in that file.
		If path is to a directory, loads all objects saved in that directory.
		
		When used from a script, this function is best used as follows::
		globals().update(load(path))
		
	Returns:
		Dictionary of loaded objects, as {name: obj ...}
	"""
	dic = {}
	if isfile(path):
		name = splitext(basename(path))[0]
		dic[name] = pickle.load(open(path, 'rb'))
	elif isdir(path):
		os.chdir(path)
		for f in os.listdir('.'):
			name = splitext(basename(f))[0]
			dic[name] = pickle.load(open(f, 'rb'))
		os.chdir('..')
	else:
		print "Could not load any data from desired path."
	return dic
	
	