from datetime import datetime
from airflow import DAG
from eai_graph_tools.airflow_data.dag_components.prediction_unb2017 import create_grid_prediction_dag

cfg_name = "refex_rolx_agg_gs_fext_deg_dim3_interval10"

dag = DAG(dag_id=cfg_name,
          default_args={'start_date': datetime.utcfromtimestamp(0)},
          start_date=datetime.utcfromtimestamp(0),
          schedule_interval=None)

create_grid_prediction_dag(dag,
                           cfg_file="eai_graph_tools/airflow_data/configs/configs_unb2017.ini",
                           cfg_name=cfg_name)
