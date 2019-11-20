import os
from eai_graph_tools.airflow_data.dag_components.canonical import get_setup_experiment, \
    get_fit_predict_xgboost, \
    load_config_file, \
    get_create_interval_metrics_tabular_xgboost


def create_tabular_prediction_dag(dag,
                                  cfg_file,
                                  cfg_name):

    WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']
    load_config_file(WD + cfg_file)

    setup_experiment = get_setup_experiment(dag, cfg_name)
    local_outlier_factor_fit_predict = get_fit_predict_xgboost(dag,
                                                               cfg_name,
                                                               force_exec=True)
    create_interval_metrics = get_create_interval_metrics_tabular_xgboost(dag,
                                                                          cfg_name,
                                                                          force_exec=True)

    setup_experiment >> local_outlier_factor_fit_predict
    local_outlier_factor_fit_predict >> create_interval_metrics
