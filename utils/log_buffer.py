# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List, Optional, Sequence
from collections import defaultdict

import torch
import numpy as np


class LogBuffer:
    def __init__(self):
        self._val_history = defaultdict(HistoryBuffer)
        self._output = {}

    @property
    def values(self):
        return self._val_history

    @property
    def output(self):
        return self._output

    def clear(self):
        self._val_history.clear()
        self.clear_output()

    def clear_output(self):
        self._output.clear()

    def update(self, vars):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            scalar_type = (int, float, torch.Tensor, np.ndarray)
            if isinstance(var, Sequence) and len(var) == 2:
                var = list(var) # change tuple
                var[0] = float(var[0])
                self._val_history[key].update(*var)
            elif isinstance(var, scalar_type):
                var = float(var)
                self._val_history[key].update(var)
            else:
                raise TypeError(f"var must be a Sequence with length of 2, "
                                f"int, float, ndarray or Tensor scalar, "
                                f"but got {type(var)}")

    def average(self, n=0):
        r"""Average latest n values or all values."""
        assert n >= 0
        for key in self._val_history:
            self._output[key] = self._val_history[key].average(n)

    def get(self, name):
        r"""Get the values of name"""
        return self._val_history.get(name, None)

    @property
    def avg(self):
        avg_dict = {}
        for key in self._val_history:
            avg_dict[key] = self._val_history[key].avg
        return avg_dict
        
    @property
    def latest(self):
        latest_dict = {}
        for key in self._val_history:
            latest_dict[key] = self._val_history[key].latest
        return latest_dict


class HistoryBuffer:
    r"""
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._values: List[float] = []
        self._nums: List[float] = []
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, num: Optional[float] = None) -> None:
        r"""
        Add a new scalar value and the number of counter. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        self._values.append(value)
        if num is None:
            num = 1
        self._nums.append(num)

        self._count += 1
        self._sum = sum(map(lambda x: x[0] * x[1], zip(self._values, self._nums)))
        self._global_avg = self._sum / sum(self._nums)

    def median(self, window_size: int) -> float:
        r"""
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median(self._values[-window_size:])

    def average(self, window_size: int) -> float:
        r"""
        Return the mean of the latest `window_size` values in the buffer.
        """
        _sum = sum(map(lambda x: x[0] * x[1], zip(self._values[-window_size:],
                                                  self._nums[-window_size:])))
        return _sum / sum(self._nums[-window_size:])

    @property
    def latest(self) -> float:
        r"""
        Return the latest scalar value added to the buffer.
        """
        return self._values[-1]

    @property
    def avg(self) -> float:
        r"""
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    @property
    def values(self) -> List[float]:
        r"""
        Returns:
            number: content of the current buffer.
        """
        return self._values

    @property
    def nums(self) -> List[float]:
        r"""
        Returns:
            number: content of the current buffer.
        """
        return self._nums

