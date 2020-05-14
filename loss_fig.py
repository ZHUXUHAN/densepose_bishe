import numpy as np
import matplotlib.pyplot as plt
import math


def l1(value):
    return abs(value)


def smoothl1(value):
    if abs(value) < 1:
        return 0.5 * math.pow(value, 2)
    else:
        return abs(value) - 0.5


def l2(value):
    return 0.5 * pow(value, 2)


def lg2(value):
    if abs(value) < 1:
        return 0.5 * math.log(1 + abs(value))
    else:
        k = 0.5/(1+abs(value))
        return k*abs(value) + (0.5*math.log(2)-1/4)


def l4(value):
    if abs(value) < 2:
        return 0.5 * math.pow(value, 8)
    else:
        return 4*abs(value) - 3.5


def twox(value):
    if abs(value) < 1:
        return 0.5*math.pow(2, abs(value)) - 0.5
    else:
        return 2*math.log(2)*abs(value)+ 1/2 -2*math.log(2)


def l3(value):
    if abs(value) < 1:
        return 0.5 * math.pow(abs(value), 3)
    else:
        return 3/2*abs(value) - 1


def l25(value):
    a = 1
    if abs(value) < a:
        return 0.5 * math.pow(abs(value), 5/2)
    else:
        return 5/4*abs(value) - 3/4*a


def l32(value):
    # return 0.5 * math.pow(abs(value), abs(3 - abs(value)))
    if abs(value) < 1:
        return 0.5 * math.pow(abs(value), 3/2)
    else:
        return 3/4*abs(value) - 1/4


def l12(value):
    # return 0.5 * math.pow(abs(value), abs(3 - abs(value)))
    if abs(value) < 1:
        return 0.5 * math.pow(abs(value), 5/2)
    else:
        return 5/2*abs(value) -2


def xx(value):

    # return math.pow(abs(value), 2 / math.log(1 + abs(value), 2))
    if abs(value) < 1:
       return 0.5 * math.pow(abs(value), math.log(2 + abs(value)))
    else:
        k = math.log(2 + abs(value))*math.log(abs(value))*(1+math.log(abs(value)))
        return k*abs(value) + 0.5


font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
plt.figure(figsize=(16, 16))
plt.title("UV-Loss", fontsize=10)
x = np.linspace(-3, 3, 2000)
y1 = np.array([]) #l1
y2 = np.array([]) #l2
y3 = np.array([]) #smoothl1
y4 = np.array([]) #lg2
y5 = np.array([]) #l4
y6 = np.array([]) #twox
y7 = np.array([]) #l3
y8 = np.array([]) #l25
y9 = np.array([]) #l32
y10 = np.array([]) #l12
y11 = np.array([]) #lxx
for v in x:
    y1 = np.append(y1, np.linspace(l1(v), l1(v), 1))
    y2 = np.append(y2, np.linspace(l2(v), l2(v), 1))
    y5 = np.append(y5, np.linspace(l4(v), l4(v), 1))
    y7 = np.append(y7, np.linspace(l3(v), l3(v), 1))
    y4 = np.append(y4, np.linspace(lg2(v), lg2(v), 1))
    y3 = np.append(y3, np.linspace(smoothl1(v), smoothl1(v), 1))
    y6 = np.append(y6, np.linspace(twox(v), twox(v), 1))
    y8 = np.append(y8, np.linspace(l25(v), l25(v), 1))
    y9 = np.append(y9, np.linspace(l32(v), l32(v), 1))
    y10 = np.append(y10, np.linspace(l12(v), l12(v), 1))
    y11 = np.append(y11, np.linspace(xx(v), xx(v), 1))
# plt.plot(x, y1, 'b', ls='-')
# plt.plot(x, y2, 'g', ls='-')
# plt.plot(x, y5, 'gray', ls='-')
# plt.plot(x, y4, 'brown', ls='-')
# plt.plot(x, y7, 'cornflowerblue', ls='-')
# plt.plot(x, y6, 'steelblue', ls='-')
plt.plot(x, y3, 'r', ls='-')
plt.plot(x, y8, 'gold', ls='-')
# plt.plot(x, y10, 'gold', ls='-')
# plt.plot(x, y11, 'g', ls='-')
plt.legend((u'L1', u'L2', u'L4', u'Lg2', 'L3', 'sqrt2', u'SmoothL1'), loc='lower right', prop=font)

plt.savefig('loss.png')
plt.show()
