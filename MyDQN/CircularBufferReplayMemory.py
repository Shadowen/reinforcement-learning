import random


class CircularBufferReplayMemory():
    """
    A simple circular buffer implemented on an array. It act like an array up until max_length, at which point it
    will begin overwriting old elements.
    Sampling occurs uniformly at random without replacement.
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.current = 0
        self.current_length = 0

        self.array = [None] * self.max_length;

    def append(self, transition):
        self.array[self.current] = transition
        self.current = (self.current + 1) % self.max_length
        self.current_length = max(self.current, self.current_length)

    def sample(self, batch_size):
        if self.current_length < self.max_length:
            return random.sample(self.array[:self.current_length], batch_size)
        return random.sample(self.array, batch_size)

    def __len__(self):
        return self.current_length
