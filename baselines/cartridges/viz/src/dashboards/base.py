from dataclasses import dataclass, field
import json
import os
import tempfile
import time
from typing import TYPE_CHECKING, Callable
import pandas as pd


import wandb
from wandb.apis.public.runs import Run


@dataclass
class PlotSpec:
    df: pd.DataFrame
    id: str
    plot_name: str
    x_col: str
    y_col: str

@dataclass
class TableSpec:
    run: Run
    path: str
    step: int

    score_col: str = "score"
    answer_col: str = "answer"
    prompt_col: str = "prompt"
    pred_col: str = "pred"

    def materialize(
        self,
    ) -> pd.DataFrame:
        temp_dir = tempfile.mkdtemp()

        try:
            # Download the file to the temporary directory
            file = self.run.file(self.path)
            file.download(temp_dir, replace=True)
            
            # Read the downloaded file as JSON and convert to DataFrame
            with open(os.path.join(temp_dir, self.path), "r") as f:
                table_data = json.load(f)
            
            df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
        finally:
            # Clean up the temporary directory and its contents
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        return df


@dataclass
class Slice:
    name: str
    metrics: dict[str, float]
    df: pd.DataFrame

SliceFn = Callable[[pd.DataFrame], list[Slice]]
@dataclass
class Dashboard:
    name: str

    # wandb filters to apply for this dashboard
    filters: dict[str, dict]

    # name of the table to use for the dashboard
    table: str = "generate_codehop/table"

    score_metric: str = "generate_codehop/score"
    step: str = "train/optimizer_step"

    slice_fns: list[SliceFn] = field(default_factory=list)

    def slices(self, df: pd.DataFrame) -> list[Slice]:
        slices = []
        for slice_fn in self.slice_fns:
            slices.extend(slice_fn(df))
        return slices

    def tables(
        self,
        run: "wandb.Run",
    ) -> list[TableSpec]:
        t0 = time.time()
        
        history = list(run.scan_history(
            keys=[self.table, self.step],
        ))
        tables = []
        for row in history:
            if row[self.table] is not None:
                tables.append(TableSpec(run=run, path=row[self.table]["path"], step=row[self.step]))

        print(f"Loading {len(tables)} tables took {time.time() - t0} seconds")

        return tables
         
    
    def plots(
        self,
        run: "wandb.Run",
    ) -> list[PlotSpec]:
        t0 = time.time()
        df = pd.DataFrame([
            {
                self.step: row[self.step],
                self.score_metric: row[self.score_metric],
            }
            for row in run.scan_history(keys=[self.step, self.score_metric])
        ])
     
        print(f"Loading {len(df)}plot rows took {time.time() - t0} seconds")
        return [
            PlotSpec(
                df=df,
                id=run.id,
                plot_name=run.name,
                x_col=self.step,
                y_col=self.score_metric,
            )
        ]


class DashboardRegistry:

    def __init__(self):
        self.dashboards = {}

    def register(self, dashboard: Dashboard):
        self.dashboards[dashboard.name] = dashboard
        
registry = DashboardRegistry()