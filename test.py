import sys

from tqdm import tqdm
from time import sleep

''''
for i in range(10):
    pbar = tqdm(range(4))
    pbar.set_description(f"Epoch {i}")
    for char in range(4):
        pbar.set_postfix({'loss':char})
        pbar.update(1
'''
for i in range(20):
        x = 0
        pbar = tqdm(None,smoothing=0,total=40,desc=f"Epoch {0}" ,disable=False,leave=True,dynamic_ncols=True,initial=0,position=0)
        pbar.reset(40)
        for char in range(40):

                pbar.set_description(f"Epoch {i}")
                x = x + 1
                pbar.set_postfix({'loss':char})
                pbar.update()
        pbar.refresh()
        sleep(3)
        print(x)
