from pyfluo import TimeSeries
import unittest
import numpy as np

class TestTS(unittest.TestCase):

    def setUp(self):
        self.ts = TimeSeries(np.random.random((10,19)))
    
    def test_normalize(self):
        min0 = np.argmin(self.ts.data, axis=1)
        n = self.ts.normalize(minmax=(2.8,9.1))
        self.assertTrue( all([i==2.8 for i in np.min(n.data,axis=1)]) )
        self.assertTrue(all([i==9.1 for i in np.max(n.data,axis=1)]))
        self.assertTrue(np.all(min0 ==  np.argmin(n.data, axis=1)))

        amax = np.argmax(self.ts.data)
        amin = np.argmin(self.ts.data)
        n = self.ts.normalize(minmax=(-5.,-1.), by_series=False)
        self.assertTrue(np.min(n.data) == -5.)
        self.assertTrue( amin == np.argmin(n.data) )
        self.assertTrue(np.max(n.data) == -1.)
        self.assertTrue( amax == np.argmax(n.data) )

if __name__ == '__main__':
    unittest.main()
