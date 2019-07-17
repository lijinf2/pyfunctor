import random
import transform as transformer
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

    def sample_rows(self, dataset, num_samples):
        assert(num_samples < len(dataset))
        idx_set = self.distinct_ints(num_samples, 0, len(dataset) - 1)
        idx_set = list(idx_set)
        idx_set.sort()
        result = transformer.map_func(idx_set, lambda idx : dataset[idx])
        return result
