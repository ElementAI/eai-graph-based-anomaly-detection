import os
from eai_graph_tools.airflow_data.dag_components.canonical import get_setup_experiment, \
    get_fit_predict_random_forest_also, \
    load_config_file, \
    get_create_interval_metrics_tabular_random_forest_also


def create_tabular_prediction_dag(dag,
                                  cfg_file,
                                  cfg_name):

    WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']
    load_config_file(WD + cfg_file)

    setup_experiment = get_setup_experiment(dag, cfg_name)
    fit_predict = get_fit_predict_random_forest_also(dag,
                                                     cfg_name,
                                                     force_exec=True)
    create_interval_metrics = get_create_interval_metrics_tabular_random_forest_also(dag,
                                                                                     cfg_name,
                                                                                     force_exec=True)

    setup_experiment >> fit_predict
    fit_predict >> create_interval_metrics
