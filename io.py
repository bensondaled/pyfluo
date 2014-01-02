import time
import os
import pickle
from os.path import isfile, isdir, join, basename, splitext

def pickle_save(obj, file_name):
	f = open(file_name, 'wb')
	pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	f.close()

def save(items=None, dir_name=None):
	"""Save any number of objects in the current workspace, one file per object.
	
	Args:
		items (list/dict): the current workspace variables to be saved, as {'var_name': var} (Note that this accepts locals() or globals() as well, to save the entire workspace.) If type list, it is converted to a dictionary where the variables names are the name attribute of the object.
		dir_name (str): name of directory in which to save object files, defaults to a time string.
	"""	
	if dir_name == None:
		 dir_name = time.strftime("%Y%m%d_%H%M%S")

	if type(items) == list:
		try:
			items = {item.name:item for item in items}
		except AttributeError:
			print "One or more objects given has no name attribute and name not supplied."
			return
	elif type(items)!=dict:
		raise Exception('Items input not understood.')
		
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
	
	