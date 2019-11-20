import attr
import pandas as pd
import typing
from eai_graph_tools.airflow_tasks.legacy_pipeline.step import pipeline_step


@attr.s(auto_attribs=True)
class IntervalCreationCfg():
    start: pd.Timestamp
    end: pd.Timestamp
    width: pd.Timedelta = attr.ib()
    overlap: pd.Timedelta = attr.ib()

    @width.validator
    def check(self, attribute, value):
        if value > self.end - self.start:
            raise ValueError("Width must be smaller than end-start")


@attr.s(auto_attribs=True)
class Interval():
    start: pd.Timestamp
    end: pd.Timestamp


@attr.s(auto_attribs=True)
class IntervalList():
    value: typing.List[Interval]

    def __iter__(self):
        return self.value.__iter__()


# Intervals Creation

@pipeline_step(IntervalCreationCfg, IntervalList)
def create_intervals(cfg):
    intervals = []
    current_timestamp = cfg.start
    while current_timestamp < (cfg.end - cfg.width):
        intervals.append(Interval(current_timestamp, current_timestamp + cfg.width))
        current_timestamp += (cfg.width - cfg.overlap)
    if current_timestamp < cfg.end:
        intervals.append(Interval(current_timestamp, cfg.end))
    return IntervalList(value=intervals)


def get_df_intervals(df, interval_list):
    data_intervals = []
    for i in interval_list:
        start = i.start.timestamp()
        end = i.end.timestamp()
        data_intervals.append(df.loc[start:end])

    return data_intervals
