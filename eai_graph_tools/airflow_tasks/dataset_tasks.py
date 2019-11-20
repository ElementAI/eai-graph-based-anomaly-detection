import attr
import pandas as pd
import typing
from hashlib import md5
import os
import pickle
from airflow.models import Variable


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

    def save(self, path):
        pickle.dump(self.value, open(path, "wb"))
        return True, path

    def load(self, path):
        if os.path.isfile(path) is False:
            return False
        else:
            self.value = pickle.load(open(path, "rb"))
        return True


def create_intervals_param_loader(dict_dict, **kwargs):

    args_subset = {}
    for category_key in dict_dict:
        args_subset[category_key] = {}
        for item_key in dict_dict[category_key]:
            args_subset[category_key][item_key] = kwargs['params'][category_key][item_key]

    return args_subset, md5(pickle.dumps(args_subset)).hexdigest()


def create_intervals(tp, in_files, out_files, *op_args, **op_kwargs):
    """
        Calculating training/inference intervals
        Outputs a pickle file containing a list of intervals, with 'start' and 'end' values
    """
    intervals = []
    current_timestamp = tp['start']
    while current_timestamp < (tp['end'] - tp['interval_width']):
        intervals.append(Interval(current_timestamp, current_timestamp + tp['interval_width']))
        current_timestamp += (tp['interval_width'] - tp['interval_overlap'])
    if current_timestamp < tp['end']:
        intervals.append(Interval(current_timestamp, tp['end']))

    state, path = IntervalList(value=intervals).save(out_files['intervals_file'].path)

    Variable.set(tp['airflow_vars']['intervals_count'], len(intervals))

    return 'succeeded' if state is True else "failed"
