class csr(object):
    def __init__(self, dim, csrRowPtr, csrColIdx, csrData):
        '''CSR Data Struncture for Sparse Matrix
        '''
        self.name = 'csr'
        self.dim = dim
        self.csrRowPtr = csrRowPtr
        self.csrColIdx = csrColIdx
        self.csrData = csrData


class jds(object):
    def __init__(self, dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData):
        '''JDS Data Struncture for Sparse Matrix
        '''
        self.name = 'jds'
        self.dim = dim
        self.jdsRowPerm = jdsRowPerm
        self.jdsRowNNZ = jdsRowNNZ
        self.jdsColStartIdx = jdsColStartIdx
        self.jdsColIdx = jdsColIdx
        self.jdsData = jdsData


class butcher_tableau(object):
    def __init__(self, c, b, A):
        self.c = c
        self.b = b
        self.A = A