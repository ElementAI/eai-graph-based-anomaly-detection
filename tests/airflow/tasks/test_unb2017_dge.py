from datetime import datetime
from airflow.models import TaskInstance
from airflow.models import Variable
from airflow import DAG
from eai_graph_tools.airflow_data.dag_components.prediction_unb2017 import create_grid_prediction_dag
import pytest

cfg_name = "unit_test_deep_graph_embeddings_agg_gs_fext_deg_dim10_interval10"

dag = DAG(dag_id=cfg_name,
          default_args={'start_date': datetime.utcfromtimestamp(0)},
          start_date=datetime.utcfromtimestamp(0),
          schedule_interval=None)

create_grid_prediction_dag(dag,
                           cfg_file="eai_graph_tools/airflow_data/configs/configs_unb2017.ini",
                           cfg_name=cfg_name)

create_training_dataset = dag.get_task("create_training_dataset")
create_inference_dataset = dag.get_task("create_inference_dataset")
train_graph_model = dag.get_task("train_graph_model")
create_graph_model_node_embeddings = dag.get_task("create_graph_model_node_embeddings")
infer_predictions = dag.get_task("predict")
create_interval_metrics = dag.get_task("create_interval_metrics")


def overwrite_out_dir_param(path, cfg_name=""):
    Variable.set(cfg_name + 'out_dir', path)


def get_out_dir(cfg_name=""):
    return Variable.get(cfg_name + 'out_dir', default_var='')


@pytest.fixture(scope="session", autouse=True)
def setup_output_path(tmp_path_factory):
    overwrite_out_dir_param(tmp_path_factory.getbasetemp())


def test_create_training_dataset_task():
    ti = TaskInstance(task=create_training_dataset, execution_date=datetime.now())
    result = create_training_dataset.execute(ti.get_template_context())
    assert result == "succeeded"


def test_create_inference_dataset_task():
    ti = TaskInstance(task=create_inference_dataset, execution_date=datetime.now())
    result = create_inference_dataset.execute(ti.get_template_context())
    assert result == "succeeded"


def test_train_graph_model_task():
    ti = TaskInstance(task=train_graph_model, execution_date=datetime.now())
    result = train_graph_model.execute(ti.get_template_context())
    assert result == "succeeded"


def test_infer_graph_model():
    ti = TaskInstance(task=create_graph_model_node_embeddings, execution_date=datetime.now())
    result = create_graph_model_node_embeddings.execute(ti.get_template_context())
    assert result == "succeeded"


def test_infer_predictions():
    ti = TaskInstance(task=infer_predictions, execution_date=datetime.now())
    result = infer_predictions.execute(ti.get_template_context())
    assert result == "succeeded"


def test_create_interval_metrics():
    ti = TaskInstance(task=create_interval_metrics, execution_date=datetime.now())
    result = create_interval_metrics.execute(ti.get_template_context())
    print("---")
    assert result == "succeeded"
