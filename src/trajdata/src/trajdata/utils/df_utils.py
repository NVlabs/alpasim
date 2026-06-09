# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Callable, Optional

import numpy as np
import pandas as pd


def downsample_multi_index_df(
    df: pd.DataFrame, downsample_dt_factor: int
) -> pd.DataFrame:
    """
    Downsamples MultiIndex dataframe, assuming level=1 of the index
    corresponds to the scene timestep.
    """
    subsampled_df = df.groupby(level=0).apply(
        lambda g: g.reset_index(level=0, drop=True)
        .iloc[::downsample_dt_factor]
        .rename(index=lambda ts: ts // downsample_dt_factor)
    )

    return subsampled_df


def upsample_ts_index_df(
    df: pd.DataFrame,
    upsample_dt_factor: int,
    method: str,
    preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    """
    Upsamples a time indexed dataframe, applying specified method.
    Calls preprocess and postprocess before and after upsampling repsectively.

    If original data is at frames 2,3,4,5, and upsample_dt_factor is 3, then
    the original data will live at frames 6,9,12,15, and new data will
    be generated according to method for frames 7,8, 10,11, 13,14 (frames after the last frame are not generated)
    """
    if preprocess:
        df = preprocess(df)

    # first, we multiply ts index by upsample factor
    df = df.rename(index=lambda ts: ts * upsample_dt_factor)

    # get the index by adding the number of frames needed per original index
    new_index = pd.Index(
        (df.index.to_numpy()[:, None] + np.arange(upsample_dt_factor)).flatten()[
            : -(upsample_dt_factor - 1)
        ],
        name=df.index.name,
    )

    # reindex and interpolate according to method
    df = df.reindex(new_index).interpolate(method=method, limit_area="inside")

    if postprocess:
        df = postprocess(df)

    return df


def upsample_multi_index_df(
    df: pd.DataFrame,
    upsample_dt_factor: int,
    method: str,
    preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    return df.groupby(level=[0]).apply(
        lambda g: upsample_ts_index_df(
            g.reset_index(level=[0], drop=True),
            upsample_dt_factor,
            method,
            preprocess,
            postprocess,
        )
    )


def interpolate_multi_index_df(
    df: pd.DataFrame, data_dt: float, desired_dt: float, method: str = "linear"
) -> pd.DataFrame:
    """
    Interpolates the given dataframe indexed with (elem_id, scene_ts)
    where scene_ts corresponds to timesteps with increment data_dt to a new
    desired_dt.
    """
    upsample_dt_ratio: float = data_dt / desired_dt
    downsample_dt_ratio: float = desired_dt / data_dt
    if not upsample_dt_ratio.is_integer() and not downsample_dt_ratio.is_integer():
        raise ValueError(
            f"Data's dt of {data_dt}s "
            f"is not integer divisible by the desired dt {desired_dt}s."
        )

    if upsample_dt_ratio >= 1:
        return upsample_multi_index_df(df, int(upsample_dt_ratio), method)
    elif downsample_dt_ratio >= 1:
        return downsample_multi_index_df(df, int(downsample_dt_ratio))
