from sampler import Sampler
import transform as transformer
def split_by_ratio_func(dataset, ratio, random_seed = 0):
    assert(ratio > 0)
    assert(ratio < 1)
    first_size = ratio * len(dataset)

    sampler = Sampler(random_seed)
    idx_set = sampler.distinct_ints(first_size, 0, len(dataset) - 1)

    first_idx = list(idx_set)
    first_idx.sort(key = lambda n : n)

    sampled_dataset = transformer.map_func(first_idx, lambda idx : dataset[idx])


    second_idx = transformer.filter_func(range(len(dataset)), lambda idx : idx not in idx_set)
    rest_dataset = transformer.map_func(second_idx, lambda idx : dataset[idx])
    
    return [sampled_dataset, rest_dataset]

