import time
import os
import pickle
from os.path import isfile, isdir, join, basename, splitext

def save(items=None, dir_name=None):
	"""Save any number of objects in the current workspace, one file per object.
	
	Args:
		items (list): the current workspace variables to be saved, defaults to all public objects in workspace.
		dir_name (str): name of directory in which to save object files, defaults to a time string.
	
	Note:
		This function does not implement the saving of objects, but rather calls upon object's *save()* method. Thus, if an object does not have a defined *save()*, this function will raise an exception.
	"""
	if dir_name == None:
		 dir_name = time.strftime("%Y%m%d_%H%M%S")
	if items == None:
		items = {i:globals(i) for i in globals().keys() if i[0] != '_'}
	elif type(items)==list:
		items = {i:globals(i) for i in globals().keys() if globals()[i] in items}
	elif type(items)==dict:
		pass
	else:
		raise Exception('Items input not understood.')
	
	if dir_name not in os.listdir('.') or not isdir(dir_name):
		os.system('mkdir %s'%dir_name)
	os.chdir(dir_name)
	for item in items:
		try:
			os.system('pwd')
			items[item].save(item)
		except AttributeError:
			print "Could not save item: %s"%item
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
	
	