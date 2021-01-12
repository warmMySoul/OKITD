import numpy as np

def dist(A, B):
    diff = (A - B)**2
    return np.sqrt(sum(diff.ravel())) 

# Нахождение Хеммингова расстояния 
def xemmingdist(A, B):
    result = (A - B)
    return np.sum(abs(result.ravel()))

def class_of_each_point(X, centers):
  m = len(X)
  k = len(centers)

  distances = np.zeros((m, k))
  distances[i, j] = [xemmingdist(centers[j], X[i]) for i in range(m) [for j in range(k)]]
  return np.argmin(distances, axis=1)


def clasterize(k,X, centers):
    m = X.shape[0]
    n = X.shape[1]

    _centers = centers

    for i in range(k):
        sub_X = X[curr_iteration == i,:]
        if len(sub_X) > 0:
            _centers[i,:] = np.mean(sub_X, axis=0)

    curr_iteration = class_of_each_point(X, _centers)

    a = 0
    for i in range(len(curr_iteration)):
        a += dist(X[i], _centers[curr_iteration[i]])**2
    a = np.mean(a)

    return centers, a

def kmeans(k, X):
  while True:
     centers = clasterize(k,X)
     if check(X,centers)==True:
         break;
  return centers

def check(X, centers):
    for i in range(centers.shape[0]):
        for j in range(centers.shape[1]):
            if (np.min(X[:,j], axis=0)>centers[i,j]) or (np.max(X[:,j],axis=0)<centers[i,j]):
                return False
    return True

# Функция по заданию комментария
def start(times, k, X):
    centres = np.zeros((times + 1, 2))
    centers[i] = np.random.random((k,X.shape[1]))
    mistake = np.zeroes((times, 1))
    for i in range( times ):
        centres[i + 1], mistake[i] = clasterize(k, X, centres[i])
    return centres[np.argmin(mistake)]

