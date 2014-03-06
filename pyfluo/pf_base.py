import time as pytime

class pfBase(object):
    def __init__(self):
        self.name = self.__class__.__name__ + pytime.strftime("-%Y%m%d_%H%M%S")
