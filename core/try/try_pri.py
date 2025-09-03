import time
import sys
from tqdm import tqdm


class Status:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        print(self.msg, end="", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\r" + " " * len(self.msg), end="\r", flush=True)


with Status("Running task prediction..."):
    time.sleep(2)


for i in tqdm(range(10)):
    pass
