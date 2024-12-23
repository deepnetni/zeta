import re

a = "abc123456"
matched = re.search("[0-9]*([a-z]*)([0-9]*)", a)
print(matched.group(2))
if matched:
    matched.group(0)
    matched.group(1)  # 123
    matched.group(2)  # abc
    matched.group(3)  # 456
else:
    print("not matched return None")
