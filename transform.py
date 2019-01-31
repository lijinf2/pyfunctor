def filter_func(dataset, bool_func):
    new_dicts = []
    for row in dataset:
        if bool_func(row):
            new_dicts.append(row)
    return new_dicts


def map_func(dataset, functor):
    values = []
    for row in dataset:
        values.append(functor(row))
    return values;

def reducebykey_func(dataset, aggregator, key_extractor = lambda row: row[0], value_extractor = lambda row: row[1]): 
    pairs = {}
    for row in dataset:
        key = key_extractor(row)
        value = value_extractor(row)
        if key in pairs:
            pairs[key] = aggregator(pairs[key], value)
        else:
            pairs[key] = value
    
    return pairs;

def reduce_func(dataset, aggregator) :
    result = dataset[0]
    for i in range(len(dataset)) : 
        if i != 0 : 
            result = aggregator(result, dataset[i])
            
    return result

def joinbykey_func(left_dataset, right_dataset, left_key = lambda left_record : left_record[0], right_key = lambda right_record: right_record[0], left_value = lambda left_record: left_record[1], right_value = lambda right_record : right_record[1]):
    dt = {}
    for right_record in right_dataset:
        key = right_key(right_record)
        assert(right_key(right_record) not in dt)
        dt[key] = right_value(right_record)
        
    result_dataset = []    
    for left_record in left_dataset:
        key = left_key(left_record)
        result = (key, left_value(left_record), dt[key])
        result_dataset.append(result)
        
    return result_dataset

def groupbykey_func(dataset, key_extractor = lambda row: row[0], value_extractor = lambda row: row[1]):                                
    pairs = {}                                                                                                                                 
    for row in dataset:                                                                                                                        
        key = key_extractor(row)                                                                                                               
        value = value_extractor(row) 
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(value)                                                                                         
    return pairs;   

# every functor generates an array
def flatmap_func(dataset, functor) :
    results = []
    for row in dataset:
        array = functor(row)
        for e in array:
            results.append(e)
    return results

def indexleft_func(dataset):
    result = []
    for i in range(len(dataset)):
        result.append((i, dataset[i]))
    return result

def indexright_func(dataset):
    result = []
    for i in range(len(dataset)):
        result.append((dataset[i], i))
    return result


def select_func(dataset, functor):
    for row in dataset:
        if functor(row):
            return row
    return []

def first(dataset):
    print(dataset[0])

def print_rows(dataset, topk = -1):
    if topk == -1:
        topk = len(dataset)
    for i in range(topk):
        print(dataset[i])
        print("\n")
