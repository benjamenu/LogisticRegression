import math

def net_input(x, w):
    net_vector = []
    for i in range(len(x)):
        result = 0
        for j in range(len(x[i])):
            result += (x[i][j] * w[j])
        net_vector.append(result)

    #net_vector = [result for i in range(len(x)) for j in range(len(x[i])) result += x[i][j] * w[j]]
    return net_vector
#x = [[1, 2], [3, 4], [-1, 2.5]]
#w = [2, 0.5]
#print(net_input(x, w))

def sigmoid(z):
    return ([1 / (1 + math.e ** -i) for i in z])
#print(sigmoid([-100, 100]))

def logr_predict_proba(x, y):
    return sigmoid(net_input(x, y))
#x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
#w = [-1, -2, 1]
#print(logr_predict_proba(x, w))

def logr_predict(x, w):
    return [1 if i >= 0.5 else 0 for i in logr_predict_proba(x, w)]
'''
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
w = [1, 1, -1, 1, -2]
print(logr_predict_proba(x, w))
'''

def logr_cost(x, y, w):
    n = len(y)
    result = 0

    for i in range(n):
        result += y[i] * math.log(logr_predict_proba([x[i]], w)[0]) + (1 - y[i]) * (math.log(1 - logr_predict_proba([x[i]], w)[0]))
    return result * (-1 / n)

#x  = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
#y = [0, 1, 0, 1]
#w = [-1, -2, 1]
#print(logr_cost(x, y, w))

def logr_gradient(x, y, w):
    n = len(y)
    g_v = []
    temp = []
    result = 0

    for i in range(n):
        temp.append(y[i] - logr_predict_proba([x[i]], w)[0])

    for j in range(len(x)):
        for k in range(len(x[i])):
            result += temp[j] * -x[j][k]
        g_v.append((1 / n) * result)
        result = 0
    return g_v
x  = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
y = [0, 1, 0, 1]
w = [-1, -2, 1]
print(logr_gradient(x, y, w))
