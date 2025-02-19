# from collections import defaultdict
import numpy as np

# lst = [{"a": 123}, {"a": 456}, {"b": 789}]

dic = {}
# for _ in lst:
#     for k, v in _.items():
#         dic.setdefault(k, []).append(v)

dic.update({"a": 1, "b": 2})
print(dic, "@")
dic.update({"a": None, "b": 2})
print(dic, "@")
# print([{k: v} for k, v in dic.items()])

# a = {"c": 2, "d": 3}
# a.update({"e": 4})
# print(a)


# a = np.log(-1.4)
# b = np.nan_to_num(a)
# print(b)


def f(d):
    d.update({"3": 4})


if __name__ == "__main__":
    a = {"1": 1, "2": 2}
    f(a)
    print(a)
