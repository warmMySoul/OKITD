import numpy as np

# Разница массивов.
def AbsArrayDifference(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    if(A.shape != B.shape):
        raise

    return np.absolute(A - B)

# Генерация массивов.
def AddMass(arrayAmount, itemAmmount)
    arrays = np.zeros((arrayAmount, itemAmmount))
    
    for i in range(arrayAmount):
        arrays[i] = 10 * np.random.random(itemAmmount) - 5      
    
    return arrays[]

# Наибольшая сумма массива.
def HighestSumArray(arrayAmount, itemAmmount):
    
    arrays = AddMass(arrayAmount, itemAmmount)
    
    sums = np.sum(arrays, axis=1)
    index = np.argmax(sums)
    
    return arrays[index]

# Евклидово расстояние.
def EuclidDistance(A, B):
    diff = (A - B)**2

    return np.sqrt(sum(diff.ravel()))

# Нахождение расстояния.
def MinkovDistance(A, B, p, r):
    result = (A - B)
    return np.sum(abs(result.ravel()) ** p) ** (1/r)
