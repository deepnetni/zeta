#!/usr/bin/env python3

from tqdm import tqdm
import time

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

pbar = tqdm(range(10), colour="blue")

for i in pbar:
    time.sleep(1)
    # pbar.set_postfix_str(f"{GREEN}hello{RESET}")
    # pbar.set_postfix({"status": f"{GREEN}Processing{RESET}", "iteration": i + 1})
    show = dict(a=10, b=12, c=123)
    pbar.set_postfix_str(", ".join(f"{GREEN}{k}={v}{RESET}" for k, v in show.items()))
