import re
import os


text = "clns1_train_2232_tl-21_fileid_158.wav"

pattern = r"fileid_[a-zA-Z0-9]+.wav"
match = re.findall(pattern, text)
print(match)

pattern = r"fileid_[a-zA-Z0-9]+.wav"
pattern = r"fileid_(\w+).wav"
match = re.search(pattern, text)
if match:
    print(match.group())
else:
    print("#")

pattern = r".*fileid_(\w+).wav"
match = re.search(pattern, text)
if match:
    print(match.group(1))
    out = re.sub(pattern, r"clean_fileid_\1", text)
    print(out)
else:
    print("#")


print(os.path.exists(None))
