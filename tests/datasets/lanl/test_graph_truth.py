# flake8: noqa
import pandas as pd

boston_x = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 1.0], [3.0, 3.0, 3.0, 1.0], [3.0, 3.0, 3.0, 1.0], [3.0, 3.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
boston_edges = [[3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], [4, 5, 6, 3, 5, 6, 3, 4, 6, 3, 4, 5]]
sse_edges = [[3, 3, 4, 4, 5, 5, 6, 6], [4, 5, 3, 6, 3, 6, 4, 5]]
degree_x = [[0.0], [0.0], [0.0], [488.0], [488.0], [488.0], [488.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
y = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

occurence_true_keys = ['ANONYMOUS LOGON@C586', 'C1250', 'C586', 'userdomain_source', 'userdomain_destination', 'computer_source', 'computer_destination', 'C101$@DOM1', 'C988', 'C1020$@DOM1', 'C1020', 'SYSTEM@C1020', 'C1021$@DOM1', 'C1021', 'C625', 'C1035$@DOM1', 'C1035', 'C1069$@DOM1', 'C1069', 'SYSTEM@C1069', 'C1085$@DOM1', 'C1085', 'C612', 'C1151$@DOM1', 'C1151', 'SYSTEM@C1151', 'C1154$@DOM1', 'C1154', 'SYSTEM@C1154', 'C1164$@DOM1', 'C119$@DOM1', 'C119', 'C528', 'C1218$@DOM1', 'C1218', 'C529', 'C1235$@DOM1', 'C1241$@DOM1', 'C1241', 'SYSTEM@C1241', 'C1250$@DOM1', 'C1314$@DOM1', 'C1314', 'C467', 'C144$@DOM1', 'C144', 'SYSTEM@C144', 'C1444$@DOM1', 'C1444', 'C1492$@DOM1', 'C1492', 'C1798', 'C1497$@DOM1', 'C1504$@DOM1', 'C1504', 'U45@C1504', 'C1543$@DOM1', 'C1543', 'SYSTEM@C1543', 'C1559$@DOM1', 'C1678$@DOM1', 'C1065', 'C457', 'C1727$@DOM1', 'C1727', 'C1881', 'C2516', 'C175$@DOM1', 'C175', 'SYSTEM@C175', 'C1767$@DOM1', 'C1767', 'U86@DOM1', 'C1920$@DOM1', 'C1920', 'C1934$@DOM1', 'C1934', 'SYSTEM@C1934', 'C1975$@DOM1', 'C1975', 'SYSTEM@C1975', 'C2067$@DOM1', 'C2067', 'SYSTEM@C2067', 'C2095$@DOM1', 'C2095', 'SYSTEM@C2095', 'C210$@DOM1', 'C210', 'SYSTEM@C210', 'C2129$@DOM1', 'C2155$@DOM1', 'C2230$@DOM1', 'C2288$@DOM1', 'C235$@DOM1', 'C2434$@DOM1', 'C2434', 'SYSTEM@C2434', 'C2622$@DOM1', 'C2622', 'SYSTEM@C2622', 'C2648$@DOM1', 'C2648', 'SYSTEM@C2648', 'C269$@DOM1', 'C2694$@DOM1', 'C2695', 'C2700$@DOM1', 'C2700', 'SYSTEM@C2700', 'C28$@DOM1', 'C28', 'SYSTEM@C28', 'C282$@DOM1', 'C282', 'SYSTEM@C282', 'C3$@DOM1', 'C3', 'SYSTEM@C3', 'C316$@DOM1', 'C316', 'SYSTEM@C316', 'C3236$@DOM1', 'C3236', 'SYSTEM@C3236', 'C3327$@DOM1', 'C3327', 'SYSTEM@C3327', 'C3450$@DOM1', 'C3450', 'C4352$@DOM1', 'C457$@DOM1', 'C516', 'C467$@DOM1', 'C2485', 'U188@DOM1', 'C495$@DOM1', 'C495', 'SYSTEM@C495', 'C520$@DOM1', 'C520', 'SYSTEM@C520', 'C528$@DOM1', 'U147@DOM1', 'C538$@DOM1', 'C553$@DOM1', 'C553', 'U175@DOM1', 'C555$@DOM1', 'C555', 'SYSTEM@C555', 'C567$@DOM1', 'C101', 'C574', 'C580$@DOM1', 'C580', 'SYSTEM@C580', 'C599$@DOM1', 'C600$@DOM1', 'C600', 'SYSTEM@C600', 'C608$@DOM1', 'C608', 'C625$@DOM1', 'C2052', 'C653$@DOM1', 'C653', 'SYSTEM@C653', 'C660$@DOM1', 'C660', 'SYSTEM@C660', 'C674$@DOM1', 'C674', 'SYSTEM@C674', 'C688$@DOM1', 'C688', 'C693$@DOM1', 'C725$@DOM1', 'C725', 'SYSTEM@C725', 'C740$@DOM1', 'C740', 'SYSTEM@C740', 'C748$@DOM1', 'C748', 'SYSTEM@C748', 'C784$@DOM1', 'C784', 'C810$@DOM1', 'C820$@DOM1', 'C820', 'SYSTEM@C820', 'C90$@DOM1', 'C90', 'SYSTEM@C90', 'C916$@DOM1', 'C922$@DOM1', 'C977$@DOM1', 'C977', 'SYSTEM@C977', 'C988$@DOM1', 'LOCAL SERVICE@C2493', 'C2493', 'U101@DOM1', 'C1862', 'C1862$@DOM1', 'U10@DOM1', 'C229', 'C62', 'U1137@DOM1', 'U119@DOM1', 'U129@DOM1', 'C419', 'C2191', 'U15@DOM1', 'C1709', 'C1708', 'C1932', 'C1931', 'C2093', 'C2092', 'U1782@DOM1', 'C2800', 'U198@DOM1', 'C1484', 'U1@DOM1', 'C456', 'U20@DOM1', 'C1750', 'U20', 'C716', 'U21@DOM1', 'C1603', 'U25@DOM1', 'U22@DOM1', 'C965', 'U23@DOM1', 'C1720', 'U24@DOM1', 'U2483@DOM1', 'C1922', 'U25@DOM3', 'C798', 'U30@DOM3', 'U26@DOM1', 'C1730', 'TGT', 'U26', 'C616', 'U27@DOM1', 'U30@DOM1', 'U327@DOM1', 'U32@DOM1', 'C815', 'U3@DOM1', 'C1191', 'C626', 'U415@DOM1', 'C1570', 'U46@DOM1', 'C423', 'U47@DOM1', 'C1152', 'U48@DOM1', 'U4@DOM1', 'U59@DOM1', 'C1634', 'U5@DOM1', 'U66@DOM1', 'C3868', 'U68@DOM1', 'C1681', 'U2@DOM1', 'C1679', 'U6@DOM1', 'C1183', 'C606', 'U6', 'C61', 'C92', 'U7@DOM1', 'U73@?', 'C1692', 'U77@DOM1', 'C2742', 'U78@DOM1', 'C1848', 'U81@C2547', 'C2547', 'C2547$@DOM1', 'U86@?', 'C1654', 'C1846', 'U898@DOM1', 'C2944', 'U8@DOM1', 'U90@DOM1', 'C1785', 'C1785$@DOM1', 'C1786', 'C1786$@DOM1', 'U91@DOM1', 'C1787', 'C1787$@DOM1']
interval_true = {'timestamp': 1, 'timestamp_original': pd.Timestamp('1970-01-01 00:00:01'), 'timestamp_adjusted': pd.Timestamp('1970-01-01 00:00:01'), 'ANONYMOUS LOGON@C586': False, 'C1250': False, 'C586': False, 'C101$@DOM1': False, 'C988': False, 'C1020$@DOM1': False, 'C1020': False, 'SYSTEM@C1020': False, 'C1021$@DOM1': False, 'C1021': False, 'C625': False, 'C1035$@DOM1': False, 'C1035': False, 'C1069$@DOM1': False, 'C1069': False, 'SYSTEM@C1069': False, 'C1085$@DOM1': False, 'C1085': False, 'C612': False, 'C1151$@DOM1': False, 'C1151': False, 'SYSTEM@C1151': False, 'C1154$@DOM1': False, 'C1154': False, 'SYSTEM@C1154': False, 'C1164$@DOM1': False, 'C119$@DOM1': False, 'C119': False, 'C528': False, 'C1218$@DOM1': False, 'C1218': False, 'C529': False, 'C1235$@DOM1': False, 'C1241$@DOM1': False, 'C1241': False, 'SYSTEM@C1241': False, 'C1250$@DOM1': False, 'C1314$@DOM1': False, 'C1314': False, 'C467': False, 'C144$@DOM1': False, 'C144': False, 'SYSTEM@C144': False, 'C1444$@DOM1': False, 'C1444': False, 'C1492$@DOM1': False, 'C1492': False, 'C1798': False, 'C1497$@DOM1': False, 'C1504$@DOM1': False, 'C1504': False, 'U45@C1504': False, 'C1543$@DOM1': False, 'C1543': False, 'SYSTEM@C1543': False, 'C1559$@DOM1': False, 'C1678$@DOM1': False, 'C1065': False, 'C457': False, 'C1727$@DOM1': False, 'C1727': False, 'C1881': False, 'C2516': False, 'C175$@DOM1': False, 'C175': False, 'SYSTEM@C175': False, 'C1767$@DOM1': False, 'C1767': False, 'U86@DOM1': False, 'C1920$@DOM1': False, 'C1920': False, 'C1934$@DOM1': False, 'C1934': False, 'SYSTEM@C1934': False, 'C1975$@DOM1': False, 'C1975': False, 'SYSTEM@C1975': False, 'C2067$@DOM1': False, 'C2067': False, 'SYSTEM@C2067': False, 'C2095$@DOM1': False, 'C2095': False, 'SYSTEM@C2095': False, 'C210$@DOM1': False, 'C210': False, 'SYSTEM@C210': False, 'C2129$@DOM1': False, 'C2155$@DOM1': False, 'C2230$@DOM1': False, 'C2288$@DOM1': False, 'C235$@DOM1': False, 'C2434$@DOM1': False, 'C2434': False, 'SYSTEM@C2434': False, 'C2622$@DOM1': False, 'C2622': False, 'SYSTEM@C2622': False, 'C2648$@DOM1': False, 'C2648': False, 'SYSTEM@C2648': False, 'C269$@DOM1': False, 'C2694$@DOM1': False, 'C2695': False, 'C2700$@DOM1': False, 'C2700': False, 'SYSTEM@C2700': False, 'C28$@DOM1': False, 'C28': False, 'SYSTEM@C28': False, 'C282$@DOM1': False, 'C282': False, 'SYSTEM@C282': False, 'C3$@DOM1': False, 'C3': False, 'SYSTEM@C3': False, 'C316$@DOM1': False, 'C316': False, 'SYSTEM@C316': False, 'C3236$@DOM1': False, 'C3236': False, 'SYSTEM@C3236': False, 'C3327$@DOM1': False, 'C3327': False, 'SYSTEM@C3327': False, 'C3450$@DOM1': False, 'C3450': False, 'C4352$@DOM1': False, 'C457$@DOM1': False, 'C516': False, 'C467$@DOM1': False, 'C2485': False, 'U188@DOM1': False, 'C495$@DOM1': False, 'C495': False, 'SYSTEM@C495': False, 'C520$@DOM1': False, 'C520': False, 'SYSTEM@C520': False, 'C528$@DOM1': False, 'U147@DOM1': False, 'C538$@DOM1': False, 'C553$@DOM1': False, 'C553': False, 'U175@DOM1': False, 'C555$@DOM1': False, 'C555': False, 'SYSTEM@C555': False, 'C567$@DOM1': False, 'C101': False, 'C574': False, 'C580$@DOM1': False, 'C580': False, 'SYSTEM@C580': False, 'C599$@DOM1': False, 'C600$@DOM1': False, 'C600': False, 'SYSTEM@C600': False, 'C608$@DOM1': False, 'C608': False, 'C625$@DOM1': False, 'C2052': False, 'C653$@DOM1': False, 'C653': False, 'SYSTEM@C653': False, 'C660$@DOM1': False, 'C660': False, 'SYSTEM@C660': False, 'C674$@DOM1': False, 'C674': False, 'SYSTEM@C674': False, 'C688$@DOM1': False, 'C688': False, 'C693$@DOM1': False, 'C725$@DOM1': False, 'C725': False, 'SYSTEM@C725': False, 'C740$@DOM1': False, 'C740': False, 'SYSTEM@C740': False, 'C748$@DOM1': False, 'C748': False, 'SYSTEM@C748': False, 'C784$@DOM1': False, 'C784': False, 'C810$@DOM1': False, 'C820$@DOM1': False, 'C820': False, 'SYSTEM@C820': False, 'C90$@DOM1': False, 'C90': False, 'SYSTEM@C90': False, 'C916$@DOM1': False, 'C922$@DOM1': False, 'C977$@DOM1': False, 'C977': False, 'SYSTEM@C977': False, 'C988$@DOM1': False, 'LOCAL SERVICE@C2493': False, 'C2493': False, 'U101@DOM1': False, 'C1862': False, 'C1862$@DOM1': False, 'U10@DOM1': False, 'C229': False, 'C62': False, 'U1137@DOM1': False, 'U119@DOM1': False, 'U129@DOM1': False, 'C419': False, 'C2191': False, 'U15@DOM1': False, 'C1709': False, 'C1708': False, 'C1932': False, 'C1931': False, 'C2093': False, 'C2092': False, 'U1782@DOM1': False, 'C2800': False, 'U198@DOM1': False, 'C1484': False, 'U1@DOM1': False, 'C456': False, 'U20@DOM1': False, 'C1750': False, 'U20': False, 'C716': False, 'U21@DOM1': False, 'C1603': False, 'U25@DOM1': False, 'U22@DOM1': False, 'C965': False, 'U23@DOM1': False, 'C1720': False, 'U24@DOM1': False, 'U2483@DOM1': False, 'C1922': False, 'U25@DOM3': False, 'C798': False, 'U30@DOM3': False, 'U26@DOM1': False, 'C1730': False, 'TGT': False, 'U26': False, 'C616': False, 'U27@DOM1': False, 'U30@DOM1': False, 'U327@DOM1': False, 'U32@DOM1': False, 'C815': False, 'U3@DOM1': False, 'C1191': False, 'C626': False, 'U415@DOM1': False, 'C1570': False, 'U46@DOM1': False, 'C423': False, 'U47@DOM1': False, 'C1152': False, 'U48@DOM1': False, 'U4@DOM1': False, 'U59@DOM1': False, 'C1634': False, 'U5@DOM1': False, 'U66@DOM1': False, 'C3868': False, 'U68@DOM1': False, 'C1681': False, 'U2@DOM1': False, 'C1679': False, 'U6@DOM1': False, 'C1183': False, 'C606': False, 'U6': False, 'C61': False, 'C92': False, 'U7@DOM1': False, 'U73@?': False, 'C1692': False, 'U77@DOM1': False, 'C2742': False, 'U78@DOM1': False, 'C1848': False, 'U81@C2547': False, 'C2547': False, 'C2547$@DOM1': False, 'U86@?': False, 'C1654': False, 'C1846': False, 'U898@DOM1': False, 'C2944': False, 'U8@DOM1': False, 'U90@DOM1': False, 'C1785': False, 'C1785$@DOM1': False, 'C1786': False, 'C1786$@DOM1': False, 'U91@DOM1': False, 'C1787': False, 'C1787$@DOM1': False}