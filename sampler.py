import random
class Sampler:
    def __init__(self, seed = 0):
        random.seed(seed)

    def distinct_ints(self, num_ints = 1, min_value = 0, max_value = 1):
        assert(max_value - min_value + 1 >= num_ints)
        # random.seed(seed)
        result = set()
        while len(result) < num_ints:
            number = random.randint(min_value, max_value)
            if number not in result:
                result.add(number)
        return result  
