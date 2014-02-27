from tifffile import TiffFile
import numpy as np

CHANNEL_IMG = 0
CHANNEL_STIM = 1

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
