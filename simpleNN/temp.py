# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


#print( [n for n in range(1, 101)] )

#print( [n**2 for n in range(1, 101)] )

print( [n**2 for n in range(1, 101) if n**2 % 3 == 0] )


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



