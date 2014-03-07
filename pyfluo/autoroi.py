import numpy as np

def cov_matrix(data):
    """
    Takes data in the following form: rows represent dimensions (x1, x2, x3 ...) and columns represent data vectors
    """
    return data.dot(data.transpose())/np.shape(data)[1]

def pca(data):
    """
    Takes data in the format used by Movie class - i.e. axis 0 = image i, axis 1 = height, axis 2 =  width
    """
    data = data.transpose([1,2,0]) # change images axis from 0 to 2
    shape = np.shape(data)
    data = data.reshape( (shape[0]*shape[1], shape[2] ) )
    data = np.matrix(data)
    data -= np.mean(data, axis=1)
    y = data.T / np.sqrt(np.shape(data)[1]-1)
    u,s,PC = np.linalg.svd(y)
    print PC
    print np.shape(PC)
    newdata = PC.T.dot(data)
    newdata = np.squeeze(np.asarray(newdata))
    newdata = newdata.reshape(shape)
    newdata = newdata.transpose([2,0,1])
    return newdata

if __name__ == '__main__':
    x = np.random.random((10,128,64))
    q=pca(x)
