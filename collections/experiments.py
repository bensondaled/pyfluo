from fluorescence import FluoSeriesCollection as FSC

class Experiment():
	def __init__(self, movie, stim_series):
		self.movie = movie
		self.stim_series = stim_series
	def generate_data(self):
		fluo_series = self.movie.extract_fluo_series()
		fsc = FSC(fluo_series=fluo_series, stim_series=self.stim_series)
		return fsc

if __name__ == "__main__":	
	pass