from functools import partial
import wandb
import pandas as pd
from src.dashboards.base import Dashboard, registry, Slice


def slice_by_depth(df: pd.DataFrame) -> list[Slice]:
    dfs = df.groupby("call_chain_depth")

    return [
        Slice(
            name=f"Call Chain Depth {depth}", 
            df=df,
            metrics={
                f"score": df["score"].mean(),
            }
        )
        for depth, df in dfs
    ]

def slice_by_file(df: pd.DataFrame) -> list[Slice]:
    dfs = df.groupby("file_name")
    return [
        Slice(
            name=f"File {file}", 
            df=df,
            metrics={
                f"score": df["score"].mean(),  
            }
        )
        for file, df in dfs
    ]


def slice_by_file_level(df: pd.DataFrame) -> list[Slice]:
    dfs = df.groupby("file_level")
    return [
        Slice(
            name=f"File Level {file}", 
            df=df,
            metrics={
                f"score": df["score"].mean(),  
            }
        )
        for file, df in dfs
    ]


for table, name in [
    ("generate_codehop", "CodeHop"),
    ("generate_codehop_w_ctx", "CodeHop with Context"),
    ("generate_codehop_w_target_level2", "CodeHop_w_Target_Level2"),

]:
    registry.register(Dashboard(
        name=f"{name}",
        filters={"$and": [{"tags": "codehop"}, {"tags": "train"}]},
        table=f"{table}/table",
        score_metric=f"{table}/score",
        step="_step",
        slice_fns=[
            slice_by_depth,
            slice_by_file_level,
        ]
    ))
        