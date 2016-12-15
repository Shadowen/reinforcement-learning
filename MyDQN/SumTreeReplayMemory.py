import random


class SumTreeReplayMemory():
    """
    Prioritized replay memory.
    Append still behaves as a circular buffer.
    Sample will sample n elements with replacement with probability proportional to TD-errors.
    Update updates previously sampled transitions with new TD-errors
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.sums_size = 2 * max_size - 1
        self.array = [None for x in range(max_size)]
        self.sums = [0 for x in range(self.sums_size)]
        self.current_index = 0

    def update_one(self, idx, key):
        """Update the sum tree with a new key for the value at a given index.
        """
        # Find the index in the sum array corresponding to the index in the payload array
        sum_idx = idx + self.max_size - 1
        self.sums[sum_idx] = key

        # Go to the left side of each pair
        sum_idx = (sum_idx - 1) // 2 * 2 + 1
        # Repeat for each node, going up the tree until we reach the root
        # Set each node's value to the sum of the left and right children
        while sum_idx != 0:
            next_idx = (sum_idx - 1) // 2
            self.sums[next_idx] = self.sums[sum_idx] + self.sums[sum_idx + 1]
            sum_idx = next_idx

    def update(self, indices, keys):
        """Uses previously obtained indices from sample() to update the keys in the sum tree structure.
        """
        for i, k in zip(indices, keys):
            self.update_one(i, k)

    def append(self, key, val):
        """Add a (key, value) pair into the sum tree, replacing the oldest tuple if the tree is full.
        """
        self.array[self.current_index] = val
        self.update_one(self.current_index, key)
        self.current_index = (self.current_index + 1) % self.max_size

    def sample_one(self):
        target = random.uniform(0, self.sums[0])
        idx = 0
        next_idx = 1
        while next_idx < self.sums_size:
            if target < self.sums[next_idx]:
                idx = next_idx
            else:
                target -= self.sums[next_idx]
                idx = next_idx + 1
            next_idx = idx * 2 + 1
        array_idx = idx - (self.max_size - 1)
        return array_idx, self.array[array_idx]

    def sample(self, count=32):
        """Samples from the sum tree count times with replacement.
        Returns a tuple of (indices, samples).
        The indices can be used to update the sum tree with new keys.
        """
        return list(zip(*(self.sample_one() for _ in range(count))))

    def __str__(self):
        return str(self.array) + '\n' + str(self.sums)

if __name__ == '__main__':
    t = SumTreeReplayMemory(4)
    t.append(1, 'a')
    t.append(1, 'b')
    t.append(4, 'c')
    t.append(6, 'd')
    t.append(8, 'e')

    print(str(t))

    from collections import defaultdict

    d = defaultdict(int)
    for x in range(10000):
        s = t.sample_one()
        d[s] += 1

    print(d)

    print(t.sample(count=3))