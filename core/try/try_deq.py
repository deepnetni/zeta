from collections import deque


dq = deque(maxlen=10)
dq.append(10)  # 0, -3
dq.append(20)  # 1, -2
dq.append(30)  # 2, -1

print(dq[-1])


# for i in range(-10, 0):
#     print(i)
#     print(dq[i])


a = "1_a_b_c"
print(a.rsplit("_", 2))
