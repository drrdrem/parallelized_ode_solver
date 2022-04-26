from random import randint as rand

def generateCSRMatrix(dim):
    MAX_NNZ_PER_ROW = int(dim//10) + 1
    csrRowPtr = [0 for _ in range(dim + 1)]
    for rowIdx  in range(dim):
        rowNNZ = rand(0, MAX_NNZ_PER_ROW)
        csrRowPtr[rowIdx + 1] = csrRowPtr[rowIdx] + rowNNZ

    NNZ = csrRowPtr[dim]
    csrColIdx = [rand(0, dim-1) for _ in range(NNZ)]
    csrData = [rand(0, 100)/100.00 for _ in range(NNZ)]

    return dim, csrRowPtr, csrColIdx, csrData


def quicksort(data, key, start, end):
    if (end - start + 1) > 1:
        left, right = start, end
        pivot = key[right]

        while left <= right:
            while key[left] > pivot:
                left += 1
            while key[right] < pivot:
                right -= 1
            
            if left <= right:
                key[left],  key[right] =  key[right], key[left]
                data[left], data[right] = data[right], data[left]

                left += 1
                right -= 1
        data, key = quicksort(data, key, start, right)
        data, key = quicksort(data, key, left, end)
    
    return data, key


def csr2jds(dim, csrRowPtr, csrColIdx, csrData):
    # Row Permutation Vector
    jdsRowPerm = [rowIdx for rowIdx in range(dim)]
    # Number of non-zeros per row
    jdsRowNNZ = [int(csrRowPtr[rowIdx + 1] - csrRowPtr[rowIdx]) for rowIdx in range(dim)]

    jdsRowPerm, jdsRowNNZ = quicksort(jdsRowPerm, jdsRowNNZ, 0, dim - 1)

    # Starting point of each compressed column
    maxRowNNZ = jdsRowNNZ[0] # Largest number of non-zeros per row
    jdsColStartIdx = [0 for _ in range(maxRowNNZ)]
    for col in range(maxRowNNZ - 1):
        # Count the number of rows with entries in this column
        cnt = 0
        for idx in range(dim):
            if(jdsRowNNZ[idx] > col):
                cnt += 1
        jdsColStartIdx[col + 1] = jdsColStartIdx[col] + cnt

    # Sort the column indexes and data
    NNZ = csrRowPtr[dim]
    jdsColIdx = [0 for _ in range(NNZ)]
    jdsData = [0. for _ in range(NNZ)]
    for idx in range(dim): # For every row
        row = jdsRowPerm[idx]
        rowNNZ = jdsRowNNZ[idx]
        for nnzIdx in range(rowNNZ):
            jdsPos = jdsColStartIdx[nnzIdx] + idx
            csrPos = csrRowPtr[row] + nnzIdx
            jdsColIdx[jdsPos] = csrColIdx[csrPos]
            jdsData[jdsPos] = csrData[csrPos]

    return dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData


def csr2matrix(dim, csrRowPtr, csrColIdx, csrData):
    matrix = [[0. for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(csrRowPtr[i], csrRowPtr[i+1]):
            matrix[i][csrColIdx[j]] = csrData[j]

    return matrix


def matrix2csr(matrix):
    dim = len(matrix)
    csrRowPtr, csrColIdx, csrData = [], [], []
    csrRowPtr = [0]
    for i in range(dim):
        val = csrRowPtr[-1]
        for j in range(dim):
            if matrix[i][j] != 0:
                val += 1
                csrColIdx.append(j)
                csrData.append(matrix[i][j])
        csrRowPtr.append(val)
    return dim, csrRowPtr, csrColIdx, csrData


if __name__ == "__main__":
    dim, csrRowPtr, csrColIdx, csrData = generateCSRMatrix(5)
    print("CSR:")
    print(csrRowPtr)
    print(csrColIdx)
    print(csrData)
    dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData = csr2jds(dim, csrRowPtr, csrColIdx, csrData)
    print("JDS:")
    print(jdsRowPerm)
    print(jdsRowNNZ)
    print(jdsColStartIdx)
    print(jdsColIdx)
    print(jdsData)
    matrix = csr2matrix(dim, csrRowPtr, csrColIdx, csrData)
    print("csr2matrix:")
    print(matrix)
    dim, csrRowPtr, csrColIdx, csrData = matrix2csr(matrix)
    print("matrix2csr:")
    print(csrRowPtr)
    print(csrColIdx)
    print(csrData)
