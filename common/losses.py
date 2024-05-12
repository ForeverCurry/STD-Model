# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t

def pearson(forecast: t.Tensor, target: t.Tensor)-> t.float:
    """
    Pearson correlation coefficient

    forecast: Forecast values. Shape:  time
    target: Target values. Shape:  time
    return: Loss value
    """
    if not t.is_tensor(forecast):
        forecast=t.from_numpy(forecast)
    if not t.is_tensor(target):
        target = t.from_numpy(target)
    forecast = forecast-t.mean(forecast)
    target = target-t.mean(target)
    return t.sum(forecast * target) / (t.sqrt(t.sum(forecast ** 2)) * t.sqrt(t.sum(target ** 2)))

def nrmse_loss(forecast: t.Tensor, target: t.Tensor)-> t.float:
    """
    NRMSE loss

    forecast: Forecast values. Shape:  time
    target: Target values. Shape:  time
    return: Loss value
    """
    # weights = divide_no_nan(mask, target)
    if not t.is_tensor(forecast):
        forecast=t.from_numpy(forecast)
    if not t.is_tensor(target):
        target = t.from_numpy(target)
    return t.sqrt(t.mean((forecast - target)**2))/t.std(target)






