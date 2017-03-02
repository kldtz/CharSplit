import sys
import numpy
from scipy.sparse import lil_matrix,csr_matrix,hstack

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_sparse_csr(filename,array):
    numpy.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def get_commandline_prefix():
    prefix = ''
    if(len(sys.argv) > 1):
        prefix = sys.argv[-1]
    return prefix
