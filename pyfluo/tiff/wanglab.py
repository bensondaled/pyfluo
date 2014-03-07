from pyfluo.pf_base import pfBase
from tifffile import TiffFile
import numpy as np
from pyfluo.movies import Movie, LineScan
from pyfluo.util import *

CHANNEL_IMG = 0
CHANNEL_STIM = 1


class MultiChannelTiff(pfBase):
    """An object that holds multiple movie-like (from movies module) objects as channels.
    
    This class is currently exclusively for creation from WangLabScanImageTiff's. Its main goal is to circumvent the need to load a multi-channel tiff file more than once in order to attain movies from its multiple channels.
    
    Attributes:
        movies (list): list of Movie objects.
        
        name (str): a unique name generated for the object when instantiated
        
    """
    def __init__(self, raw, klass=Movie, **kwargs):
        """Initialize a MultiChannelTiff object.
        
        Args:
            raw (str / WangLabScanImageTiff / list thereof): list of movies.
            skip (list): a two-item list specifying how many frames to skip from the start (first item) and end (second item) of each movie.
            klass (class): the type of class in which the movies should be stored (currently supports pyfluo.Movie or pyfluo.LineScan)
        """
        super(MultiChannelTiff, self).__init__() 
        
        self.movies = []
        
        if type(raw) != list:
            raw = [raw]
        widgets=[' Loading tiffs:', Percentage(), Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=len(raw)).start()
        for idx,item in enumerate(raw):
            if type(item) == str:
                raw[idx] = WangLabScanImageTiff(item)
                pbar.update(idx+1)
            elif type(item) != WangLabScanImageTiff:
                raise Exception('Invalid input for movie. Should be WangLabScanImageTiff or tiff filename.')
        tiffs = raw
        pbar.finish()
                
        n_channels = tiffs[0].n_channels
        if not all([i.n_channels==n_channels for i in tiffs]):
            raise Exception('Channel number inconsistent among provided tiffs.')
        
        for ch in range(n_channels):    
            movie = None
            for item in tiffs:              
                chan = item[ch]
                mov = klass(data=chan['data'], info=chan['info'], **kwargs)
                
                if movie == None:   movie = mov
                else:   movie.append(mov)
            self.movies.append(movie)
            
    def get_channel(self, i):
        return self.movies[i]
    def __getitem__(self, i):
        return self.get_channel(i)
    def __len__(self):
        return len(self.movies)

class WangLabScanImageTiff(object):

	def __init__(self, filename):
		tiff_file = TiffFile(filename)
		pages = [page for page in tiff_file]
		
		data = [page.asarray() for page in pages]
		page_info = [self.parse_page_info(page) for page in pages]
	
		ex_info = page_info[0]
		self.n_channels = int(ex_info['state.acq.numberOfChannelsAcquire'])
		
		self.channels = self.split_channels(data, page_info)
		self.source_name = filename
	def split_channels(self, data, page_info):
		if len(data)%float(self.n_channels):
			raise('Tiff pages do not correspond properly to number of channels. Check tiff parsing.')
		
		channels = []
		for ch in range(self.n_channels):
			channel = {}
			channel['data'] = np.concatenate([[i] for i in data[ch::self.n_channels]])
			channel['info'] = page_info[ch::self.n_channels]
			channels.append(channel)
		return channels
		
	def parse_page_info(self, page):
		desc = ''.join([ch for ch in page.image_description if ord(ch)<127])
		fields = [field.split('=') for field in desc.split('\n') if len(field.split('='))>1]
		info = {}
		for field in fields:
			info[field[0]] = field[1]
		return info
	def __getitem__(self, idx):
		return self.channels[idx]
if __name__ == "__main__":
	testtif = '/Users/Benson/Desktop/5_24_2013_GR_100ms_5p_071.tif'
 	tdata = WangLabScanImageTiff(testtif)

"""
http://stackoverflow.com/questions/6686550/how-to-animate-a-time-ordered-sequence-of-matplotlib-plots

Old method of processing tiffs:

from PIL import Image

tiff_file = Image.open(filename)
img_size = [raw_tiff_file.size[1], raw_tiff_file.size[0]]
self.data = []
try:
	while 1:
		raw_tiff_file.seek(raw_tiff_file.tell()+1)
		self.data.append( np.reshape(raw_tiff_file.getdata(),img_size) )
except EOFError:
    pass
self.data = np.dstack(self.data)
"""
