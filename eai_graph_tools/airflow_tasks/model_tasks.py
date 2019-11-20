import attr
import dask
import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from airflow.exceptions import AirflowException
from eai_graph_tools.datasets.intervals import FullData
from eai_graph_tools.datasets.intervals import DatasetVendor
from typing import Any, List, Mapping, Optional, Tuple  # type: ignore
from tensorboardX import SummaryWriter
from collections import defaultdict
from eai_graph_tools.graph_tools.nn.models.infomax_sageconv import Infomax, train_infomax
from eai_graph_tools.datasets.intervals import load_dataset_vendor
from eai_graph_tools.node_embeddings.refexrolx import RefexRolx
from eai_graph_tools.airflow_tasks.legacy_pipeline.step import Step
from eai_graph_tools.airflow_tasks import get_all_nodes_from_dataset


class UNBPredict(Step):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @attr.s(auto_attribs=True)
    class Input:
        data: FullData
        metric_normalization: Optional[str]

    Output = np.ndarray


class InfomaxUNBPredict(UNBPredict):
    def _run(self, inp):
        embeddings = self.model.encoder(inp.data.x, inp.data.edge_index, inp.data.num_nodes, corrupt=False)
        return embeddings.detach().cpu().numpy()


class RandRUNBPredict(UNBPredict):
    def _run(self, inp):
        return self.model.embed_roles(inp.data, inp.metric_normalization)


@attr.s(auto_attribs=True)
class InferenceOutput():
    node_embeddings: Any = attr.ib(default=None)
    output_path: os.PathLike = attr.ib(default=".")
    filename: str = attr.ib(default='nodes_embeddings.p')

    def save(self):
        file_path = os.path.join(self.output_path, self.filename)
        pickle.dump(self.node_embeddings, open(file_path, "wb"))

    def restore(self) -> bool:
        file_path = os.path.join(self.output_path, self.filename)
        if os.path.isfile(file_path) is False:
            return False
        else:
            self.node_embeddings = pickle.load(open(file_path, "rb"))
            return True


def predict_node(data,
                 file_index,
                 predictor,
                 metric_normalization,
                 tensorboard_writer,
                 nodes_of_interest,
                 connect):
    print("Predicting generic")
    embeddings = predictor.run(UNBPredict.Input(data=data, metric_normalization=metric_normalization))
    print("Preparing output")
    tensorboard_writer.add_embedding(embeddings, global_step=file_index)   # TODO: add nodes labels to embeddings
    # Open saved info on node positions, access nodes of interest.
    node_indexes_dict = data.node_indexes_in_tensors
    print("Loaded indices")
    node_embeddings: Mapping = defaultdict(list)
    for noi in nodes_of_interest:
        if noi in node_indexes_dict and node_indexes_dict[noi] < embeddings.shape[0]:
            idx = node_indexes_dict[noi]
            noi_embeddings = embeddings[idx, ]
            node_embeddings[noi] = noi_embeddings
        else:
            node_embeddings[noi] = np.full(embeddings[0, ].shape, np.nan)
    return node_embeddings


def index_embeddings(dataset: DatasetVendor,
                     predictor: UNBPredict,
                     device: torch.device,
                     tensorboard_writer: SummaryWriter,
                     nodes_of_interest: List[str],
                     metric_normalization: Optional[str]):

    node_embeddings: Mapping = defaultdict(list)

    dask_intervals: Tuple = ()
    # Use this value to hold the future of the most recent delayed task
    # This forces every task to execute in sequence
    prev = None
    for file_index, file_path in enumerate(dataset.files):
        data = dataset.get(file_index)
        prev = dask.delayed(predict_node)(data,
                                          file_index,
                                          predictor,
                                          metric_normalization,
                                          tensorboard_writer,
                                          nodes_of_interest,
                                          prev)
        # Tuple append
        dask_intervals = dask_intervals + (prev,)
    # Unify the output with a passthrough task
    output = dask.delayed(lambda *results: results)(*dask_intervals).compute()
    for embedding in output:
        for noi in nodes_of_interest:
            node_embeddings[noi].append(embedding[noi])
    return node_embeddings


@attr.s(auto_attribs=True)
class ModelWrapper():
    model: nn.Module
    filename: str = attr.ib(default='trained_model')

    def save(self):
        torch.save(self.model.state_dict(), self.filename)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            return self.model
        else:
            raise AirflowException(f"Model is missing: {filename}")

    def restore(self) -> bool:
        if os.path.isfile(self.filename) is False:
            return False
        else:
            self.model.load_state_dict(torch.load(self.filename))
            self.model.eval()
            return True


def train_deep_graph_embeddings_model(log,
                                      in_files,
                                      out_files,
                                      start,
                                      interval_width,
                                      hidden_dim,
                                      training_epochs,
                                      tensorboard_writer,
                                      patience_epochs,
                                      learning_rate,
                                      **kwargs):

    log.info(train_deep_graph_embeddings_model)
    verbose = kwargs['verbose'] if 'verbose' in kwargs else True

    dataset = load_dataset_vendor(in_files,
                                  start,
                                  interval_width)

    input_dim = len(dataset.get(0).x[0])
    print("Desired Backend: %s" % ('cuda' if torch.cuda.is_available() else 'cpu'))
    device = torch.device('cpu')

    infomax_model = Infomax(input_dim, hidden_dim).to(device)
    infomax_optimizer = torch.optim.Adam(infomax_model.parameters(), lr=learning_rate)

    for step, interval_data in enumerate(dataset):
        data = interval_data.to(device)

        if verbose:
            start = dataset.start + step * dataset.interval_width
            end = dataset.start + (step + 1) * dataset.interval_width
            print(f"Generated a data training interval: START {start} END {end}")
            print(f"Nodes: {data.num_nodes}")
            print(f"Edges: {data.num_edges}")
            print(f"Feat.: {data.num_features}")

    if verbose:
        print(f"Starting Training ({len(dataset)} intervalscreate_graph_model_node_embeddings_hash, "
              f"{training_epochs} epochs)")

    output = ModelWrapper(model=infomax_model,
                          filename=out_files['trained_model'].path)
    output.save()

    global_count = 0
    lower_loss = 1000
    patience_epochs = patience_epochs
    epochs_stagnation_count = 0
    for epoch in range(1, training_epochs + 1):
        epochs_stagnation_count = epochs_stagnation_count + 1
        if epochs_stagnation_count == patience_epochs:
            if verbose:
                print(f"Early stop, patience={patience_epochs} exceeded. Lower loss reached: {lower_loss}")
            return True
        for step, data in enumerate(dataset):
            loss = train_infomax(infomax_model, infomax_optimizer, data, epoch)
            global_count = global_count + 1
            if loss < lower_loss:
                lower_loss = loss
                epochs_stagnation_count = 0
                output = ModelWrapper(model=infomax_model,
                                      filename=out_files['trained_model'].path)
                output.save()

                if verbose:
                    print(f"Saved model with loss: {loss}")

            if tensorboard_writer:
                tensorboard_writer.add_scalar('data/loss', float(loss), global_count)

            if verbose:
                print('Epoch: {:03d}, Step: {:03d}, Loss: {:.7f}'.format(epoch, step, loss))


def train_randr_model(log,
                      in_files,
                      out_files,
                      start,
                      interval_width,
                      hidden_dim,
                      feature_extractor):

    log.info("train_rand_model")
    dataset = load_dataset_vendor(in_files,
                                  start,
                                  interval_width)

    model = RefexRolx(hidden_dim, feature_extractor)
    model.initialize_embeddings(dataset.get(0))
    pickle.dump(model, open(out_files['trained_model'].path, 'wb'))


def train_graph_model(log,
                      in_files,
                      out_files,
                      start,
                      end,
                      interval_width,
                      hidden_dim,
                      feature_extractor,
                      training_epochs,
                      predicator_name,
                      tensorboard_writer,
                      patience_epochs,
                      learning_rate):

    print("train graph models")

    if predicator_name == "infomax":
        train_deep_graph_embeddings_model(log,
                                          in_files,
                                          out_files,
                                          start,
                                          interval_width,
                                          hidden_dim,
                                          training_epochs,
                                          tensorboard_writer,
                                          patience_epochs,
                                          learning_rate)
    elif predicator_name == "randr":
        train_randr_model(log,
                          in_files,
                          out_files,
                          start,
                          interval_width,
                          hidden_dim,
                          feature_extractor)
    return "succeeded"


"""
    INFER GRAPH MODEL
"""


def infer_graph_model(log,
                      in_files,
                      out_files,
                      start,
                      end,
                      interval_width,
                      predicator_name,
                      hidden_dim,
                      nodes_of_interest,
                      tensorboard_writer):

    log.info(f"infer_graph_model, start:{start}, end={end}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trained_model_path = in_files['trained_model'].path
    dataset_files_paths = in_files
    del dataset_files_paths['trained_model']

    dataset = load_dataset_vendor(in_files,
                                  start,
                                  interval_width)

    if len(nodes_of_interest) == 0:
        nodes_of_interest = get_all_nodes_from_dataset(dataset)

    if predicator_name == 'infomax':
        input_dim = len(dataset.get(0).x[0])
        trained_model = ModelWrapper(model=Infomax(input_dim,
                                                   hidden_dim).to(device)).load(trained_model_path)
        predictor = InfomaxUNBPredict(trained_model)

    elif predicator_name == 'randr':
        trained_model = pickle.load(open(trained_model_path, 'rb'))
        predictor = RandRUNBPredict(trained_model)
    else:
        raise AirflowException('predicator_name unsupported')

    node_embeddings = index_embeddings(dataset=dataset,
                                       predictor=predictor,
                                       device=device,
                                       tensorboard_writer=tensorboard_writer,
                                       nodes_of_interest=nodes_of_interest,
                                       metric_normalization=None)

    print(f"Saving node_embeddings to: {out_files['node_embeddings'].path}")
    output = InferenceOutput(node_embeddings=node_embeddings,
                             filename=out_files['node_embeddings'].path)

    output.save()
    return 'succeeded'
