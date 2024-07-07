""" a, b = map(int, input().split())

print(format(a/b, '.2f'))
print('{:.2f}'.format(a/b)) 
print(f'{a/b:.2f}') # Main
print('%.2f' %(a/b)) """
    
""" a = input()
a = int(a, 16)

for i in range(1, 16):
    # print("%X"%a, "*%X" %i, "=%X" %(a*i), sep='')
    # print("%X*%X=%X" %(a, i, a*i))
    print(f"{a:X}*{i:X}={a*i:X}") """
    
""" num = int(input("정수 1개를 입력하시오: "))
for x in range(1, num+1):
    if '3' in str(x) or '6' in str(x) or '9' in str(x):
    # if any([True for i in ['3', '6', '9'] if i in str(x)]): # if만
    # if any([True if i in str(x) else False for i in ['3', '6', '9']]): # if, else
    # if any((one_num in str(x)) for one_num in ['3', '6', '9']):
    # if any((one_char in ['3', '6', '9']) for one_char in str(i)):
        print("X", end=' ')
    else:
        print(x, end=' ') """
        
""" import numpy as np

print(all(np.array([20, 40, 50]) < 45)) """

""" import numpy as np

list1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

array1 = np.array(list1)
print(list(array1)) # List 안에 담긴 Array
print(array1.tolist()) # List

y = array1[:, 1] # List가 아닌 Array에서만 가능
print(list(y)) # List
print(y.tolist()) # List """

""" import numpy as np

index_array = np.array([2, 4, 5])
array1 = np.array([11, 12, 13, 14, 15, 16])

print(array1[index_array]) """

""" import numpy as np

list = [[0, 23], [1, 21], [2, 22]]
list.sort(key=lambda x: x[1])
print(list)"""

""" a = None
b = "Hello"
result = a or b
print(result)  # 출력: Hello

a = [1, 2, 3]
b = [4, 5, 6]
result = a or b
print(result)  # 출력: [1, 2, 3] """

""" for x in range(1, 11, 2):
    # print(f"{'*'*x:^11}")
    # print(f"{'*'*x:>11}")
    # print(f"{'*'*x:<11}")
    
    print(f"{'*'*x:-^11}") """
    
""" num = 0.12347
print(f"{num:.4f}") # 0.1235

print(f"{num:10.3f}")
print(f"{num:^10.3f}")
print(f"{num:<10.3f}")

print(f"{num:a>10.3f}")
print(f"{num:a^10.3f}")
print(f"{num:a<10.3f}") """