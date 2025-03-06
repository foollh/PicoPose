# Copyright (c) Facebook, Inc. and its affiliates.
# auto registry all inplace lr_scheduler
from functools import partial
from math import pi, cos
import warnings
import numpy as np
import torch
from torch.optim.lr_scheduler import *

import math
from bisect import bisect_right
from typing import List

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.constant_ending > 0:
            # Constant ending lr.
            if (math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                    < self.constant_ending):
                return [
                    base_lr * self.constant_ending for base_lr in self.base_lrs
                ]
        return [
            base_lr * math.pow(
                (1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


# modify from https://github.com/cmpark0126/pytorch-polynomial-lr-decay/blob/master/torch_poly_lr_decay/torch_poly_lr_decay.py

class DevPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_decay_steps: int,
                 end_learning_rate: float = 0.0001,
                 power: float = 1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps)**(self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps)**(self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class StepLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [
                group["lr"] * group.get("lr_multi", 1.0)
                for group in self.optimizer.param_groups
            ]
        return [
            group["lr"] * self.gamma * group.get("lr_multi", 1.0)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma**(self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]



class InvLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Note p as current epoch or iter number according to user setting;
    Note maxp as num_epochs or num_iters according to user setting;
    p / maxp ranges from [0, 1].
    Then lr(p) = base_lr * (1 + gamma * p / maxp)**(-power).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        maxp (int): Maximum epochs (or maximum iterations if
            lr_scheduler.step() is called each iteration)
        gamma (float): Default: 0.1.
        power (float): Default: 0.75.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> scheduler = InvLR(optimizer, gamma=10, power=0.75)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    # def __init__(self, optimizer, maxp, gamma=10, power=0.75, last_epoch=-1):
    def __init__(self, optimizer, gamma=10, power=0.75, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        # self.maxp = maxp
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        choice = 2
        if choice == 1:
            # implementation 1 (mine): fixed gamma(10), the shape of lr-iter curve will keep
            # the same although maxp are different
            progress = self.last_epoch / self.maxp
            return [
                base_lr * (1 + self.gamma * progress)**(-self.power) *
                group.get("lr_mult", 1.0) for group, base_lr in zip(
                    self.optimizer.param_groups, self.base_lrs)
            ]
        elif choice == 2:
            # implementation 2 (Long Mingsheng): fixed gamma(0.001), the same last_epoch will
            # lead to the same lr, although maxp are different
            return [
                base_lr * (1 + self.gamma * self.last_epoch)**(-self.power) *
                group.get("lr_mult", 1.0) for group, base_lr in zip(
                    self.optimizer.param_groups, self.base_lrs)
            ]


# modify from https://github.com/PRBonn/lidar-bonnetal/blob/master/train/common/warmupLR.py
# NOTE: maybe not work, need to fix

class WarmupCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    r""" Warmup learning rate scheduler.
      Initially, increases the learning rate from 0 to the final value, in a
      certain number of steps. After this number of steps, each step decreases
      LR exponentially.
  """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_lr: float,
                 base_lr: float = 0.0,
                 warmup_iters: int = 1000,
                 momentum: float = 0.9,
                 decay: float = 0.99,
                 last_epoch: int = -1):
        # cyclic params
        self.max_lr = max_lr
        self.warmup_iters = warmup_iters
        self.momentum = momentum
        self.decay = decay

        # cap to one
        if self.warmup_iters < 1:
            self.warmup_iters = 1

        # cyclic lr
        self.initial_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=self.max_lr,
            step_size_up=self.warmup_iters,
            step_size_down=self.warmup_iters,
            cycle_momentum=False,
            base_momentum=self.momentum,
            max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer, self.last_epoch)

    def get_lr(self):
        return [
            self.max_lr * (self.decay**self.last_epoch) for lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_iters:
            if not self.finished:
                self.base_lrs = [self.max_lr for lr in self.base_lrs]
                self.finished = True
            return super().step(epoch)
        else:
            return self.initial_scheduler.step(epoch)



class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}", milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor)
        return [
            base_lr * warmup_factor *
            self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()



class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        cycle_factor: float = 1.0,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        start_cos_after_warmup: bool = False,
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.cycle_factor = cycle_factor
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.start_cos_after_warmup = start_cos_after_warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor)
        if self.start_cos_after_warmup:
            if self.last_epoch <= self.warmup_iters:
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
            cosine = math.cos(math.pi * self.cycle_factor *
                              (self.last_epoch - self.warmup_iters) /
                              max(1, (self.max_iters - self.warmup_iters)))
        else:
            cosine = math.cos(math.pi * self.cycle_factor * self.last_epoch /
                              self.max_iters)

        if self.cycle_factor > 0.5:
            # scale curve to prevent negative value
            cosine = 0.5 * (1.0 + cosine)

        if not self.start_cos_after_warmup:
            cosine = warmup_factor * cosine
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [base_lr * cosine for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()



class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor)
        if self.constant_ending > 0 or warmup_factor == 1.0:
            # Constant ending lr.
            if (math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                    < self.constant_ending):
                return [
                    base_lr * self.constant_ending for base_lr in self.base_lrs
                ]
        return [
            base_lr * warmup_factor * math.pow(
                (1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int,
                               warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters or warmup_iters == 0:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        # grow linearly from `warmup_factor` to 1.0
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError(f"Unknown warmup method: {method}")


def adjust_learning_rate(optimizer,
                         epoch,
                         args,
                         mode="auto",
                         value=0.1,
                         namelist=[]):
    r"""
    Adjust the learning rate according to the epoch
    Parameters
    ----------
    optimzer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model
    epoch: int
        The current epoch
    args: Namespace
        Arguments that main.py receive
    total_epochs: int
        The total epoch number
    mode: str
        Mode of setting lr, 'auto' (computed automatically by formula), 'rel' (relative) or 'abs' (absolute)
    value: float
        In 'auto' mode, lr of pretrained modules additionally multiply this variable;
        In 'rel' mode, parameters multiply this variable;
        In 'abs' mode, parameters is set to this variable
    namelist: list
        If namelist is not empty, then only adjust the lr of param_groups whose name is in namelist;
        If namelist is empty (default), then adjust the lr of all param_group
    Return
    ------
    The function has no return
    """
    select_groups = []
    if len(namelist) == 0:
        select_groups = optimizer.param_groups
    else:
        for param_group in optimizer.param_groups:
            if param_group["type"] in namelist:
                select_groups.append(param_group)

    for param_group in select_groups:
        if mode == "auto":
            p = float(epoch) / args.epochs
            lr = args.base_lr / ((1 + 10 * p)**0.75)
            lr_pretrain = lr * value
            for param_group in optimizer.param_groups:
                if param_group["type"] == "pre-trained":
                    param_group["lr"] = lr_pretrain
                else:
                    param_group["lr"] = lr
        elif mode == "rel":
            param_group["lr"] = param_group["lr"] * value
        elif mode == "abs":
            param_group["lr"] = value



def flat_and_anneal_lr_scheduler(
    optimizer,
    total_iters,
    warmup_iters=0,
    warmup_factor=0.1,
    warmup_method="linear",
    warmup_pow=2,
    anneal_point=0.72,
    anneal_method="cosine",
    target_lr_factor=0,
    poly_power=1.0,
    step_gamma=0.1,
    steps=[2 / 3.0, 8 / 9.0],
    cyclic=False,
    return_function=False,
):
    """Ref: https://github.com/fastai/fastai/blob/master/fastai/callbacks/flat_cos_anneal.py.

    warmup_initial_lr = warmup_factor * base_lr
    target_lr = base_lr * target_lr_factor
    total_iters: cycle length; set to max_iter to get a one cycle schedule.
    """
    if warmup_method not in ("constant", "linear", "pow", "exp"):
        raise ValueError(
            "Only 'constant', 'linear', 'pow' or 'exp' warmup_method accepted," "got {}".format(warmup_method)
        )

    if anneal_method not in (
        "cosine",
        "linear",
        "poly",
        "exp",
        "step",
        "none",
    ):
        raise ValueError(
            "Only 'cosine', 'linear', 'poly', 'exp', 'step' or 'none' anneal_method accepted,"
            "got {}".format(anneal_method)
        )

    if anneal_method == "step":
        if any([_step < warmup_iters / total_iters or _step > 1 for _step in steps]):
            raise ValueError(
                "error in steps: {}. warmup_iters: {} total_iters: {}."
                "steps should be in ({},1)".format(
                    steps,
                    warmup_iters,
                    total_iters,
                    warmup_iters / total_iters,
                )
            )
        if list(steps) != sorted(steps):
            raise ValueError("steps {} is not in ascending order.".format(steps))
        warnings.warn("ignore anneal_point when using step anneal_method")
        anneal_start = steps[0] * total_iters
    else:
        if anneal_point > 1 or anneal_point < 0:
            raise ValueError("anneal_point should be in [0,1], got {}".format(anneal_point))
        anneal_start = anneal_point * total_iters

    def f(x):  # x is the iter in lr scheduler, return the lr_factor
        # the final lr is warmup_factor * base_lr
        x = x % total_iters if cyclic else x  # cyclic
        if x < warmup_iters:
            if warmup_method == "linear":
                alpha = float(x) / warmup_iters
                return (1 - warmup_factor) * alpha + warmup_factor
            elif warmup_method == "pow":
                alpha = float(x) / warmup_iters
                return (1 - warmup_factor) * pow(alpha, warmup_pow) + warmup_factor
            elif warmup_method == "exp":
                assert warmup_factor > 0, warmup_factor
                alpha = float(x) / warmup_iters
                return warmup_factor ** (1 - alpha)
            elif warmup_method == "constant":
                return warmup_factor

        if x < anneal_start:
            return 1
        elif x >= anneal_start and x < total_iters:
            if anneal_method == "step":
                # ignore anneal_point and target_lr_factor
                milestones = [_step * total_iters for _step in steps]
                lr_factor = step_gamma ** bisect_right(milestones, float(x))
            elif anneal_method == "cosine":
                # slow --> fast --> slow
                lr_factor = target_lr_factor + 0.5 * (1 - target_lr_factor) * (
                    1 + cos(pi * ((float(x) - anneal_start) / (total_iters - anneal_start)))
                )
            elif anneal_method == "linear":
                # (y-m) / (B-x) = (1-m) / (B-A)
                lr_factor = target_lr_factor + (1 - target_lr_factor) * (total_iters - float(x)) / (
                    total_iters - anneal_start
                )
            elif anneal_method == "poly":
                # slow --> fast if poly_power < 1
                # fast --> slow if poly_power > 1
                # when poly_power == 1.0, it is the same with linear
                lr_factor = (
                    target_lr_factor
                    + (1 - target_lr_factor) * ((total_iters - float(x)) / (total_iters - anneal_start)) ** poly_power
                )
            elif anneal_method == "exp":
                # fast --> slow
                # do not decay too much, especially if lr_end == 0, lr will be
                # 0 at anneal iter, so we should avoid that
                _target_lr_factor = max(target_lr_factor, 5e-3)
                lr_factor = _target_lr_factor ** ((float(x) - anneal_start) / (total_iters - anneal_start))
            else:
                lr_factor = 1
            return lr_factor
        elif x >= total_iters:
            return target_lr_factor

    if return_function:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f), f
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

