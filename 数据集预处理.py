import os
import requests
import tiktoken
import numpy
input_file_path = 'tang_poet.txt'
with open(input_file_path, 'r') as f:
    data = f.read()
