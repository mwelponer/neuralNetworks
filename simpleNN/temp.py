import numpy as np
import pandas as pd

# num = input() # wait for the user to enter something
# num = int(num)
# print(f"square of {num} is {num**2}")
        


# data = {'bananas':[3, 0, 1], 'apples':[2, None, 4], 'oranges':[2, 3, 2]}

# orders = pd.DataFrame(data)
# # print(orders)
# orders = pd.DataFrame(data, index=['Alice', 'Bob', 'Tom'])
# print(orders)
# # print(orders.loc['Bob'])
# # print(orders['apples'])

# #orders.to_json('new_orders.json')
# print(orders.mean())

# s1 = pd.Series(['a', 'b', 'c'])
# s2 = pd.Series(['d', 'e', 'f'])
# print(s1 + s2)
# print(s1 * 2)
# print(s2 > 'e')

# s3 = pd.Series({'a':1, 'b':2})
# print(s3['b'])
# print(s3[0:1])





# import time
# # define the decorator 
# def timeCount(func):
#     # inner function 
#     def inner(*args, **kwargs): # take all func arguments
#         start = time.time()
#         res = func(*args, **kwargs)
#         end = time.time()
#         print('time elapsed:', end - start)
#         return res # return the return of func

#     return inner

# # generator, yields k powers of 2 
# def pows2Gen(k):
#     i = 0
#     while i < k:
#         yield 2**i
#         i += 1

# # define main function with timeCount decorator to get elapsed time
# @timeCount
# def powsTwo(k):
#     return [n for n in pows2Gen(k)]

# # use the decorated main function
# res = powsTwo(10)
# print(res)








# a = np.array([[1, 2], 
#               [3, 4]])
# b = np.array([[5, 6], 
#               [7, 8]])


# res = np.dot(a, b)
# print(res)
# print(res.shape)

# res = a @ b
# print(res)
# print(res.shape)








# from datetime import date

# class Entity:
#     # Constructor
#     def __init__(self, name, age):
#         # to create member variables
#         self.name = name
#         self.age = age

#     # to string magic method
#     def __str__(self):
#         return f'{self.name}, {self.age} years old'
    
#     # equality magic method
#     def __eq__(self, other): 
#         if isinstance(other, Entity): 
#             # if other.name == self.name: 
#             #     return True
#             if hash(other) == hash(self):
#                 return True
#         return False
    
#     def __hash__(self):
#         # hash(custom_object)
#         return hash((self.name, self.age))

#     # private method, prefix the member name with __
#     def __calculateAge(birthYear):
#         return date.today().year - birthYear

#     @staticmethod
#     def isAdult(age): # don't use self in static methods
#         return age > 18

#     @classmethod # a sort of constructor variation
#     def createFromYear(cls, name, birthYear): # use cls to refer to Entity
#         return cls(name, cls.__calculateAge(birthYear))

# print(Entity.isAdult(23)) # use of static method
# print(Entity.isAdult(16)) # use of static method

# e1 = Entity('mike', 46) # create an instance normally
# print(e1)
# print(e1.name, e1.age)

# e2 = Entity.createFromYear('tom', 1977) # create an instance using class method
# print(e2)

# print(f'e1(id {id(e1)}) {hash(e1)}, e2(id {id(e2)}) {hash(e2)} e1 == e2: {e1 == e2}')

# e3 = Entity('mike', 46)
# print(f'e1(id {id(e1)}) {hash(e1)}, e3(id {id(e3)}) {hash(e3)} e1 == e3: {e1 == e3}')








# class Base:
#     def __init__(self, name='Alice'):
#         self.name = name
 
# class Derived(Base):
#     def __init__(self, age):
#         super().__init__()
#         self.age = age
        
 
# d = Derived(23)
# print(d.name, d.age)







# class Person:
# 	def __init__(self, name, age):
# 		self.name = name
# 		self.age = age

# class Student(Person):
# 	def __init__(self, name, age, matricola):
# 		super().__init__(name, age)
# 		self.matricola = matricola
		
# s = Student('mike', 46, 1234)
# print(s.name, s.age, s.matricola)

# arr = [i for i in range(5)]
# print(arr)

# print([[1] * 5] * 3)

# print([[0] * 2] * 3)
# print([[0] * 2 for i in range(3)])
# print([[0 for i in range(2)] for i in range(3)])






# dic = {'a':3, 'c':2, 'b':1}
# print(dic)

# ndic = {it[1]:it[0] for it in zip(dic.keys(), dic.values())}
# print(ndic)

# print(sorted(dic))
# print(sorted(dic.values()))

# # values sorted by key
# print([ dic[k] for k in sorted(dic) ])

# # keys sorted by values
# print( sorted(dic, key=dic.get))

# # print( sorted(dic, key=dic.get) )
# # print( [it[1] for it in sorted(dic.items())] )
# # [myMap[key] for key in sorted(myMap.keys())]

# characters = ["Iron Man", "Spider Man", "Captain America"]
# actors = ["Downey", "Holland", "Evans"]
# #myT = list(zip(characters, actors))






# with open('genData.py') as fp:
#     for line in fp:
#         print(line.strip())

# with open('genData.py', 'r') as file:
#     line = file.readline()
#     while line :
#         print(line.strip())
#         line = file.readline()









import heapq

a = [2, 4, 7, 1, 3]
heapq.heapify(a)

while a:
    print(heapq.heappop(a))
