import heapq
class TopK(object):
    """
    
    Attributes:
        data (list): collection contains elements
        keyer (lambda): key_extractor for item
        max_size (int): maximum heap size
        
    """
    
    def __init__(self, topk = -1, init_list = [], keyer = lambda x : x):
        self.max_size = topk
        self.keyer = keyer
        self.data = []
        
        for item in init_list:
            self.push(item)

    def push(self, item):
        heapq.heappush(self.data, (self.keyer(item), item))
        if len(self.data) > self.max_size : 
            self.pop()
        

    def pop(self):
        return heapq.heappop(self.data)[1]
    
    def get_len(self):
        return len(self.data)

    def get_data(self):
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i][1])
        return result
