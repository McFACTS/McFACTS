"""
Controls the global random state used throughout McFACTS simulations.
"""
import numpy as np
from typing import Union, Optional, Any


default_seed = 1


# UNCOMMENT this line to switch to the np.random.Generator class
# class RandomGeneratorInherited(np.random.Generator):
# COMMENT this line to switch to the Generator class
class RandomGeneratorInherited(np.random.RandomState):
    def __init__(self, seed: int):
        # UNCOMMENT these 2 lines to switch to the Generator class
        #bit_generator = np.random.PCG64(seed)
        #super().__init__(bit_generator)
        # COMMENT this line to switch to the Generator class
        super().__init__(seed)
        self.call_count = 0

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        self.call_count += 1
        return super().uniform(low, high, size)

    def shuffle(self, x: Union[np.ndarray, list]) -> None:
        self.call_count += 1
        return super().shuffle(x)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        self.call_count += 1
        return super().normal(loc, scale, size)

    def choice(self, a: Union[int, np.ndarray, list], size: Optional[Union[int, tuple]] = None, 
               replace: bool = True, p: Optional[np.ndarray] = None) -> Union[Any, np.ndarray]:
        self.call_count += 1
        return super().choice(a, size, replace, p)

    def pareto(self, a: float, size: Optional[Union[int, tuple]] = None) -> Union[float, np.ndarray]:
        self.call_count += 1
        return super().pareto(a, size)

    # UNCOMMENT to switch to the Generator class
    # the function randint changed to integers between RandomState and Generator. Either keep this wrapper function
    # or change the rng.randint calls to rng.integers
    # def randint(self, low: int, high: int = None, size: Optional[Union[int, tuple]] = None) -> Union[int, np.ndarray]:
    #     self.call_count += 1
    #     return super().integers(low, high, size)


def reset_random(seed):
    # UNCOMMENT this line to switch to the Generator class
    # rng.bit_generator.state = np.random.PCG64(seed).state
    # COMMENT this line to switch to the Generator class
    rng.seed(seed)
    rng.call_count = 0
    return rng


rng = RandomGeneratorInherited(seed=default_seed)
