import numpy as np


a = -9


def f1(d):
    return 10 ** (-d / 10)


def f2(d):
    return 0.5 ** (d / 3)


print(f1(a), f2(a))
