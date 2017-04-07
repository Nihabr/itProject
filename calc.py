import matplotlib.pyplot as plt
import numpy as np

def graph(formula, x_range, avgspeed):
    x = np.array(x_range)
    y = eval(formula)
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    plt.xlabel('Kilometers run')
    plt.ylabel('Avg Speed')
    plt.title('Breakpoint plot of walk/run speed')
    plt.plot(x, y)

    y2 = eval(avgspeed + '+ x*0')
    plt.plot(x, y2)

    plt.show()
    idx = np.argwhere(np.diff(np.sign(y - y2)) != 0).reshape(-1) + 0

    print('Intersection: x:' + str(x[idx]) + ', y: '+ str(y[idx]))

walkspeed = str(5)
runspeed = str(9)
avgspeed = str(6.27)
distance = str(42.3)

stringtoplot =  runspeed + '*x/' + distance + '+' + walkspeed + '*(' + distance +'-x)/'+ distance
print(stringtoplot)

graph(stringtoplot, range(0, 20), avgspeed)
