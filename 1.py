import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def autocorrelate(a):
  n = len(a)
  cor = []
  for i in range(n//2, n//2+n):
    a1 = a[:i+1]   if i< n else a[i-n+1:]
    a2 = a[n-i-1:] if i< n else a[:2*n-i-1]
    cor.append(np.corrcoef(a1, a2)[0, 1])
  return np.array(cor)

a = 10
# переменная
print(a)

a = [1,2,3]
# массив
print(a)

a = [[1],[1,2],[1,2,3]]
# матрица с заданными значениями
print(a)

a = np.zeros((2,3))
# матрица с нулевыми значениями
print(a)

a = np.ones((2,3))
# матрица с единичными значениями
print(a)

a = np.random.randint(2, 6, (5, 5))
# матрица со случайными целочисленными значениями
print(a)

data = np.random.normal(3, 5, 1000)
# массив из 1000 элементов c нормальным распределением, мат. ожиданием 3 и среднеквадратическим отклонением 5
print(data)

data = np.loadtxt('./test.txt', dtype=np.int32)
#импорт данных из файла и указываем что там целые числа
print(data)

#загрузка одномерных данных, ищем(узнаем) ключи, извлекаем данные
data = scipy.io.loadmat('./1D/var1.mat').keys()['n']

dataMax=np.max(data)
#расчет максимального
print(dataMax)

dataMin=np.min(data)
#расчет минимального
print(dataMin)

dataMedian=np.median(data)
#расчет медианы
print(dataMedian)

dataMean=np.mean(data)
#расчет среднего арифметического
print(dataMean)

dataVar=np.var(data)
#расчет дисперсии
print(dataVar)

dataStd=np.std(data)
#расчет среднеквадратичного отклонения
print(dataStd)

plt.plot(data)
plt.title("График") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
#график
plt.show()

#значение массива, среднее значение и дисперсия на одном графике
mean = np.mean(data) * np.ones(len(data))
var = np.var(data) * np.ones(len(data))
plt.plot(data, 'b-', mean, 'r-', mean-var, 'g--', mean+var, 'g--')
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()
plt.show()

#сетка + количество столбцов на диаграмме
plt.hist(data, bins=20)
#показываем сетку на графике
plt.grid()
plt.show()

#одномерная матрица to одномерный массив затем выполняем корреляцию и строим график
data = np.ravel(data)
cor = autocorrelate(data)
plt.plot(cor)
plt.show()


datadata = scipy.io.loadmat('./ND/var1.mat').keys()['mn']

n = datadata.shape[1]
corr_matrix = np.zeros((n,n))

for i in range(0,n):
    for j in range(0,n):
        a1 = datadata[:,i]   
        a2 = datadata[:,j]   
        corr_matrix[i,j]=np.corrcoef(a1,a2)[0,1]    

np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(datadata[:, 2], datadata[:, 5], 'b.')
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()
plt.show()
