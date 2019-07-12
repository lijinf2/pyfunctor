from threading import Thread

def map_func(dataset, functor, num_threads = 10): 
    result = [[] for i in range(num_threads)]
    chunk_size = int (len(dataset) / num_threads) + 1
    
    def thread_map_func(idx):    
        start = idx * chunk_size
        end = start + chunk_size
        end = min(end, len(dataset))

        for row in dataset[start:end]:
            result[idx].append(functor(row))

    threads = []
    for i in range(num_threads):
        threads.append(Thread(target = thread_map_func, args=(i, )))
        
    for t in threads:
        t.start()
        #t.run()
    
    for t in threads:
        t.join()
    
    final = []
    for r in result:
        final += r
    return final                                                                                                                                               
 
def test_map_func():
    dataset = [i for i in range(20000000)] 
    result = map_func(dataset, lambda row : row * 2)
    print(len(result))
    print(result[:10])
