#from processing import Thread
from multiprocessing import Process, Manager
import time

def process_map_func(procnum, data_chunk, return_dict, functor):
    result_chunk = []
    for row in data_chunk:
        result_chunk.append(functor(row))
    return_dict[procnum] = result_chunk

def get_chunk(dataset, idx, chunk_size):
    start = idx * chunk_size
    end = start + chunk_size
    end = min(end, len(dataset))
    return dataset[start:end]

def map_func(dataset, functor, num_processes = 10): 
    chunk_size = int (len(dataset) / num_processes) + 1
    
    result_dict = Manager().dict()

    jobs = []
    for i in range(num_processes):
        jobs.append(Process(target = process_map_func, args=(i, get_chunk(dataset, i, chunk_size), result_dict, functor)))
        
    for t in jobs:
        t.start()
        #t.run()
    
    for t in jobs:
        t.join()
    
    final = []
    for i in range(num_processes):
        final += result_dict[i]
    return final
 
def test_map_func():
    start_time = time.time()
    dataset = [i for i in range(100000000)] 
    result = map_func(dataset, lambda row : row * 2)
    print("time: %f" % (time.time() - start_time))
    print(len(result))
    print(result[:10])

def test_map_func_single():
    start_time = time.time()
    dataset = [i for i in range(100000000)] 
    for i in range(len(dataset)):
        dataset[i] = dataset[i] * dataset[i]
    result = dataset
    print("time: %f" % (time.time() - start_time))
    print(len(result))
    print(result[:10])


#test_map_func()
#test_map_func_single()
