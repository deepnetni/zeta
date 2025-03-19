# from collections import defaultdict
import time

import numpy as np
from tqdm import tqdm

dic = {}
# for _ in lst:
#     for k, v in _.items():
#         dic.setdefault(k, []).append(v)

dic.update({"c": 1, "b": 2})
dic.update({"c": None, "aa": 2})
# print([{k: v} for k, v in dic.items()])
print(dic.items(), "@")

# a = {"c": 2, "d": 3}
# a.update({"e": 4})
# print(a)


# a = np.log(-1.4)
# b = np.nan_to_num(a)
# print(b)


def f(d):
    d.update({"3": 4})


if __name__ == "__main__":
    a = {"a": 1, "c": 2, "b": 3}
    b = {"e": 3, "f": {"a": 11, "b": 12, "g": 13}}

    # n = dict(a, **b)
    n = dict(a, **b)
    print(n)

    g = n.get("h", {})
    print(g)
    g.update({"gg": 111})
    print(n)
