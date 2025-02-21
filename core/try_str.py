#!/usr/bin/env python3
import os

a = "a.b.c.wav"
print(os.path.splitext(a), a.removesuffix("c.wav"))
