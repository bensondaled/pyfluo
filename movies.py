from pyfluo.tiff import WangLabScanImageTiff
from pyfluo.stimulation import StimSeries
from pyfluo.time_series import TimeSeries
from pyfluo.time_series_collection import TimeSeriesCollection
import matplotlib.animation as animation
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import os
import json

class MultiChannelMovie(object):
	def __init__(self, raw, skip_beginning=0, skip_end=0):
		self.movies = []
		
		if type(raw) != list:
			raw = [raw]
		for idx,item in enumerate(raw):
			if type(item) == str:
				raw[idx] = WangLabScanImageTiff(item)
			elif type(item) != WangLabScanImageTiff:
				raise Exception('Invalid input for movie. Should be WangLabScanImageTiff or tiff filename.')
		tiffs = raw
		
		n_channels = tiffs[0].n_channels
		for ch in range(n_channels):	
			data = None
			info = []
		
			for item in tiffs:
				if item.n_channels != n_channels:
					raise Exception('Channel number inconsistent among provided tiffs.')
				
				chan = item[ch]
				dat = chan['data']
				info += (chan['info'])
				if data == None:
					data = dat
				else:
					if np.shape(dat)[:2] != np.shape(data)[:2]:
						raise Exception('Movies are not of the same dimensions.')
					data = np.append(data, dat, axis=2)
		
			mov = Movie(data=data, info=info, skip_beginning=skip_beginning, skip_end=skip_end)
			self.movies.append(mov)
			
	def get_channel(self, i):
		return self.movies[i]
	def __len__(self):
		return len(self.movies)

class Movie(object):
	def __init__(self, data, info, skip_beginning=0, skip_end=0):
			
		self.data = data
		self.info = info
			
		if skip_beginning:
			self.data = self.data[:,:,skip_beginning:]
			self.info = self.info[skip_beginning:]
		if skip_end:
			self.data = self.data[:,:,:-skip_end]
			self.info = self.info[:-skip_end]
		
		self.ex_info = self.info[0]
		lpf = float(self.ex_info['state.acq.linesPerFrame'])
		ppl = float(self.ex_info['state.acq.pixelsPerLine'])
		mspl = float(self.ex_info['state.acq.msPerLine'])
		self.pixel_duration = mspl / ppl / 1000. #seconds
		self.frame_duration = self.pixel_duration * ppl * lpf #seconds
		
		self.time = np.arange(len(self))*self.frame_duration
		self.width = np.shape(self.data)[1]
		self.height = np.shape(self.data)[0]
				
		self.rois = []
		self.roi_pts = []
	
	# Special Calls
	def __getitem__(self, idx):
		return self.data[:,:,idx]
	def __len__(self):
		return np.shape(self.data)[2]
	def __str__(self):
		return '\n'.join([
		'Movie object.',
		"Length: %i frames."%len(self),
		"Frame Dimensions: %i x %i"%(np.shape(self.data)[0], np.shape(self.data)[1]),
		"Duration: %f seconds."%(self.time[-1]-self.time[0]+self.frame_duration),
		])
	
	# Public Methods
	
	# Modifying data
	def append_movie(self, movies):
		if type(movies) == Movie:
			movies = [movies]
		if type(movies) != list:
			raise Exception('Not a valid data type to append to a Movie object (should be Movie or list of Movies).')
		for m in movies:
			if m.frame_duration != self.frame_duration:
				raise Exception('Frame rates of movies to be appended do not match.')
			self.data = np.append(self.data, m.data, axis=2)
		self.time = np.arange(len(self))*self.frame_duration
	
	# Reshaping data	
	def as_series(self):
		flat_data = np.transpose(self.data,[2,0,1]).flatten()
		t = np.arange(len(flat_data))*self.pixel_duration
		return Series1D(flat_data, time=t)		
	def as_stim_series(self):
		flat_data = np.transpose(self.data,[2,0,1]).flatten()
		t = np.arange(len(flat_data))*self.pixel_duration
		return StimSeries(flat_data, time=t)
	
	# ROI analysis
	def select_roi(self, store=True):
		zp = self.z_project(show=True, rois=True)
		mask = None
		pts = pl.ginput(0)
		if pts:
			path = mpl_path.Path(pts)
			mask = np.ones(np.shape(zp),dtype=bool)
			for ridx,row in enumerate(mask):
				for cidx,pt in enumerate(row):
					if path.contains_point([ridx,cidx]):
						mask[cidx,ridx] = False
			if store:
				self.roi_pts.append(path.vertices)
				self.rois.append(mask)
			pl.close()
		return (pts, mask)
	def extract_by_roi(self, rois=None, method=np.ma.mean):
		series = []
		if rois == None:
			rois = self.rois
		if type(rois) == int:
			rois = [self.rois[rois]]
		elif type(rois) != list:
			rois = [rois]
		for roi in rois:
			if type(roi)==int:
				roi = self.rois[roi]
			roi = np.dstack([roi for i in self])
			masked = np.ma.masked_array(self.data, mask=roi)
			ser = method(method(masked,axis=0),axis=0)
			ser = TimeSeries(ser, time=self.time)
			series.append(ser)
		series = TimeSeriesCollection(series)
		
		if len(series) == 1:
			return series[0]
		else:
			return series
	def clear_roi(self, idxs=None):
		if not idxs:
			idxs=range(len(self.rois))
		for roi in idxs:
			self.rois[roi] = None
			self.roi_pts[roi] = None
		self.rois = [r for r in self.rois if r]
		self.roi_pts = [r for r in self.roi_pts if r]
	
	# Visualizing data
	def z_project(self, method=np.mean, show=False, rois=False):
		zp = method(self.data,2)
		if show:
			pl.imshow(zp, cmap=mpl_cm.Greys_r)
			if rois:
				for roi in self.roi_pts:
					pl.scatter(roi[:,0], roi[:,1])
			pl.xlim([0, self.width])
			pl.ylim([self.height, 0])
		return zp
	def play(self, repeat=False, **kwargs):
		flag = pl.isinteractive()
		pl.ioff()
		fig = pl.figure()
		ims = [ [pl.imshow(i, cmap=mpl_cm.Greys_r)] for i in self ]
		
		ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat=repeat, **kwargs)
		pl.show()
		if flag:	pl.ion()
		
	# Saving data
	def save(self, name):
		os.system('mkdir %s'%name)
		os.system('cd %s'%name)
		
		metadata_file = open('metadata.json','w')
		metadata = {}
		metadata['source_files'] = self.source_names
		metadata['tif_data'] = self.ex_info
		metadata = json.dumps(metadata)
		metadata_file.write('%s'%metadata)
		metadata_file.close()
		
		#TO BE COMPLETED
		
		os.system('cd ..')