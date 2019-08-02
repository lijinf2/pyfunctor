import sys
import time
sys.path.insert(0, "../threads") 
import transform as transformer


def multi(row):
    return row * 2

dataset = [i for i in range(10000000)] 
result = transformer.map_func(dataset, lambda row : multi(row), 4)
print(result[:10])

result = transformer.map_func(dataset, lambda row : row * 2, 4)
print(result[:10])
