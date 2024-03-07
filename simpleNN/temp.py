import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





# se = pd.Series( [21, 54, 93], index=['fild1', 'field2', 'field3'])
# print(se.)


# df = pd.DataFrame({'x':[21, 54, 93], 'y':[8, 45, 21], 'z':[1, 12, 41]}) # automatic indices 
# print(df)


# df = pd.DataFrame([[21, 54], [76, 43], [98, 93]], columns=['fild1', 'field2'])
# print(df)


# df = pd.DataFrame({'name':['mike', 'jack'], 'age':[47, 23]})
# print( df[df['name'] == 'mike'] )






# df = pd.DataFrame({"A":[-5, 8, 12, -9, 5, 3], 
#                    "B":[-1, -4, 6, 4, 11, 3], 
#                    "C":[11, 4, -8, 7, 3, -2]}) 
# print(df)
# df['A'].clip(-2, 7, inplace=True)
# print(df)







df = pd.read_csv('pdata.csv')
print(df.head())
print(df.corr())

# df.plot()
# plt.show()

# df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
# plt.show()

df["Pulse"].plot(kind = 'hist')
plt.show() # it represents the distribution of data of a specific column













# a = np.array([[1, 2]])
# b = np.array([[5, 6], 
# 			  [7, 8]])
# c = np.array([[1, 2], [3, 4]])

# print(np.dot(a, c))
# print(np.matmul(a, b))









# datatype = [('name', 'U7'), ('age', int), ('height', int)]
# people = [
#     ('Alice', 25, 170), 
#     ('Bob', 35, 180), 
#     ('Charlie', 35, 175)
# ]
# array = np.array(people, dtype = datatype) 
# np.sort(array, order='height') # sorting based on height
# # >>> [('Alice', 25, 170) ('Charlie', 35, 175) ('Bob', 30, 180)]

# res = np.sort(array, order=('age', 'height')) # sorting first on age, then on height
# print(res)














# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5, 6], [7, 8]])
# print(np.stack((arr1, arr2), axis=-1)) # stack row wise 
# # >>> [[[1, 2], [3, 4]], 
# # >>>  [[5, 6], [7, 8]]]
# print(np.stack((arr1, arr2), axis=1)) # stack col wise 
# # >>> [[[1, 2], [5, 6]], 
# # >>>  [[3, 4], [7, 8]]]

# arr = np.array([[1, 2], [3, 4], [5, 6]])
# print(arr.flatten())







# array = np.array([[3, 10, 2], 
#                   [1, 5, 7], 
#                   [2, 7, 5]])

# res = np.sort(array, axis=0)
# print(res) 
# # >>> [[ 1  5  2]
# # >>>  [ 2  7  5]
# # >>>  [ 3 10  7]]

# res = np.sort(array, axis=1) # same as axis=-1 because -1 means last dimension
# print(res) 
# # >>> [[ 2  3 10]
# # >>>  [ 1  5  7]
# # >>>  [ 2  5  7]]







# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5, 6], [7, 8]])

# print(np.concatenate((arr1, arr2), axis=0))
# print(np.concatenate((arr1, arr2), axis=1))

# print(np.stack((arr1, arr2), axis=0)) # [ [[1, 2], [3, 4]], [[5, 6], [7, 8]] ]
# print(np.stack((arr1, arr2), axis=1)) # [ [[1, 2], [5, 6]], [[3, 4], [7, 8]] ]







# d = {2:'a', 1:'c', 3:'b'}

# # keys sorted by values
# print(sorted(d, key=lambda x : d[x]))
# # values sorted by key
# print([d[i] for i in sorted(d)])








# from datetime import date

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#         self.classname = 'person'

#     def __str__(self):
#         return f'{self.classname}: {self.name}, {self.age} years old'

#     # protected method: children can use it!
#     def _getAge(birthYear):
#         return date.today().year - birthYear
    
#     # private method: only self can use it!
#     def __somePrivate(something):
#         pass

#     @classmethod
#     def createByBirthYear(self, name, birthYear):
#         return Person(name, self._getAge(birthYear))

#     @staticmethod    
#     def great():
#         print('hello!')
    
# class Student(Person): # child class
#     def __init__(self, name, age, matricola):
#         super().__init__(name, age)
#         self.matricola = matricola
#         self.classname = 'student' # override member

#     # override method 
#     @classmethod
#     def createByBirthYear(self, name, birthYear, matricola):
#         # use protected method _getAge()
#         return Student(name, super()._getAge(birthYear), matricola)

# # person
# print(Person('jack', 18))
# print(Person.createByBirthYear('mike', 1977))
# # static methods
# Person.great()
# Student.great() # inherit from parent
# # student
# print(Student('alice', 33, 1234))
# print(Student.createByBirthYear('bob', 21, 5678))
# print(Student('alice', 33, 1234).matricola)





# l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# for i in range(len(l)):
#     print(l[i], l[~i])

# print(l)
# print(l[::-1])
# print( [l[~i] for i in range(len(l))] )

# for i in range(len(l)):
#     print(l[~i])


# print( [i for i in  range(5) if i % 2 == 0] )






# # check palindrome string
# def palindrome(str):
#     for i in range(len(str)//2):
#         if str[i] != str[~i]:
#             return False

#     return True

# print(palindrome("abba"))
# print(palindrome("kayak"))
# print(palindrome("mipiaci"))

# import time
# def dec_timeit(func):
#     def inner(*args, **kwargs):
#         start = time.time()
#         res = func(*args, **kwargs)
#         print(f'time used(ms): {time.time() - start}')
#         return res 
#     return inner

# @dec_timeit
# def palindrome(str):
#     l, r = 0, len(str)-1
#     for _ in range(len(str)//2):
#         if l >= r:
#             break
#         while str[l] == ' ' and l < r: 
#             l += 1
#         while str[r] == ' ' and r > l: 
#             r -= 1

#         if str[l] != str[r]: 
#             return False
#         l += 1
#         r -= 1
#     return True

# print(palindrome("abba"))
# print(palindrome("kayak"))
# print(palindrome("mipiaci"))
# print(palindrome('was it a car or a cat i saw'))






# d = pd.DataFrame({'name':['mike', 'jack'], 'age':[45, 29]}, index=['1st', '2nd'])
# print(d)

# # to get a dataframe column
# r = d['name'] # this becomes a Series!
# # >>> 1st  mike
# # >>> 2nd  jack

# # to get a dataframe row
# r = d.loc['2nd'] # this becomes a Series!
# # >>> name  jack
# # >>> age   29
# r = d.loc['1st', 'age'] # >>> 45

# print(r)





# print('\n' + d[d['name'].str.contains('j')]) # dataframe where name contains 'j'
# print(d[~d['name'].str.contains('c')]) # dataframe where name does not contain 'c'
# print(d[d['age'] == 29]) # dataframe where age is 29






# print(0.1 + 0.2 == 0.3)
# print(0.1 + 0.2)
# print(0.3)






# print(bin(5)) # 1 0 1
# print(bin(6)) # 0 1 1
# print(bin(2)) #   1 0
# print(~2)

# print(~5) 
# print(bin(-6))
# print(~~~~5)
# b = 0b1000
# b = -9
# print(b)
# print(bin(b))
# print(~b)














# print( [n**2 for n in range(1, 101) if n**2 % 3 == 0] )

# def parse(filePath):
#     with open(filePath) as fp:
#         res = []
#         for line in fp:
#             line = line.strip()
#             raw = line.split(',')
#             raw = [int(n) for n in raw]

#             res.append(raw)
    
#     return res

# res = parse('data.csv')
# print(res)

# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __str__(self):
#         return f'({self.x}, {self.y})'
    
# p1 = Point(2, 3)
# print(p1)

# # list of points with random coordinates
# import random

# points = []
# for _ in range(100):
#     x, y = random.uniform(1,10), random.uniform(1,10) 
#     points.append(Point(x, y))