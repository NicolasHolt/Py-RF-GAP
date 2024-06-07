import random
import math
import numpy as np

class Dataset:
    def __init__(self, seed: int | float | str | bytes | bytearray | None = None):
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)

    def donut(self):
        def generate_ring(r1, r2, n):
            points = []
            for i in range(n):
                r = self.random.uniform(r1, r2)
                theta = self.random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                points.append([x, y])
            return points
        inner = generate_ring(0, 2, 200)
        outer = generate_ring(5, 8, 200)
        labels = [1 for _ in inner] + [2 for _ in outer]
        return np.array(inner + outer), np.array(labels)


