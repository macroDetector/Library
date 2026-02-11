import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()
    
    dt = df["deltatime"]

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dist"] = np.hypot(df["dx"], df["dy"])

    df["speed"] = df["dist"] / dt

    df["acc"] = df["speed"].diff() / dt

    df["jerk"] = df["acc"].diff() / dt

    df["theta"] = np.arctan2(df["dy"], df["dx"])

    df["x0"] = df["x"]
    df["x1"] = df["x"].shift(5)
    df["x2"] = df["x"].shift(10)

    df["y0"] = df['y']
    df['y1'] = df['y'].shift(5)
    df['y2'] = df['y'].shift(10)

    df["micro_shake"] = (df["dx"].diff().abs() + df["dy"].diff().abs())
                                 
    a = np.hypot(df["x1"] - df["x2"], df["y1"] - df["y2"])
    b = np.hypot(df["x0"] - df["x1"], df["y0"] - df["y1"])
    c = np.hypot(df["x0"] - df["x2"], df["y0"] - df["y2"])
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(0, s * (s - a) * (s - b) * (s - c)))
    denominator = a * b * c
    df["curvature"] = np.where(denominator > 1e-9, (4 * area) / denominator, 0)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
