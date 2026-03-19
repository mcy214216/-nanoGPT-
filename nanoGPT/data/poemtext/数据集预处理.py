import os
import requests
import tiktoken
import numpy

# from nanoGPT.bench import train_data

input_file_path = 'tang_poet.txt'
with open(input_file_path, 'r',encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc=tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")#中文：训练集有多少个标记
print(f"val has {len(val_ids):,} tokens")#中文：验证集有多少个标记

train_ids = numpy.array(train_ids, dtype=numpy.uint16)
val_ids = numpy.array(val_ids, dtype=numpy.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

