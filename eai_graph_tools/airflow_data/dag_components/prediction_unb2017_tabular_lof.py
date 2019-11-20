import os
from eai_graph_tools.airflow_data.dag_components.canonical import get_setup_experiment, \
    get_fit_predict_local_outlier_factor, \
    load_config_file, \
    get_create_interval_metrics_tabular_local_outlier_factor


def create_tabular_prediction_dag(dag,
                                  cfg_file,
                                  cfg_name,
                                  use_smote=False):

    WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']
    load_config_file(WD + cfg_file)

    setup_experiment = get_setup_experiment(dag, cfg_name)
    local_outlier_factor_fit_predict = get_fit_predict_local_outlier_factor(dag,
                                                                            cfg_name,
                                                                            force_exec=True,
                                                                            use_smote=use_smote)

    if use_smote is False:
        create_interval_metrics = get_create_interval_metrics_tabular_local_outlier_factor(dag,
                                                                                           cfg_name,
                                                                                           force_exec=True)

    setup_experiment >> local_outlier_factor_fit_predict

    if use_smote is False:
        local_outlier_factor_fit_predict >> create_interval_metrics
