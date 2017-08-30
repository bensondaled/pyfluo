##
import pyqtgraph as pg
import multiprocessing as mp

##

##
def play_mov():
    x = np.random.random([5000,256,256])
    q = pg.image(x)
    q.play(500)
    pg.QtGui.QApplication.exec_()

##

p = mp.Process(target=play_mov)

##
p.start()

##
