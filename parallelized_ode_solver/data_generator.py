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