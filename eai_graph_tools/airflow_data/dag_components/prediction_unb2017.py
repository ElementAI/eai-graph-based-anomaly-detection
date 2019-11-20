import os
from eai_graph_tools.airflow_data.dag_components.canonical import get_create_inference_dataset, \
    get_create_training_dataset, \
    get_create_graph_model_node_embeddings, \
    get_predict, \
    get_create_interval_metrics, \
    get_setup_experiment, \
    get_train_graph_model, \
    get_node_analysis, \
    load_config_file


def create_grid_prediction_dag(dag,
                               cfg_file,
                               cfg_name,
                               skip_node_analysis=False):

    WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']
    load_config_file(WD + cfg_file)

    setup_experiment = get_setup_experiment(dag, cfg_name)
    create_training_dataset = get_create_training_dataset(dag, cfg_name, force_exec=True)
    create_inference_dataset = get_create_inference_dataset(dag, cfg_name, force_exec=True)
    train_graph_model = get_train_graph_model(dag, cfg_name, force_exec=True)
    create_graph_model_node_embeddings = get_create_graph_model_node_embeddings(dag,
                                                                                cfg_name,
                                                                                use_all_nodes=False,
                                                                                force_exec=True)
    infer_predictions = get_predict(dag, cfg_name, use_all_nodes=False, force_exec=True)
    create_interval_metrics = get_create_interval_metrics(dag, cfg_name, force_exec=True)
    if skip_node_analysis is False:
        node_analysis = get_node_analysis(dag, cfg_name, force_exec=True)

    setup_experiment >> create_training_dataset
    setup_experiment >> create_inference_dataset
    create_training_dataset >> train_graph_model
    create_inference_dataset >> train_graph_model
    train_graph_model >> create_graph_model_node_embeddings
    create_graph_model_node_embeddings >> infer_predictions
    infer_predictions >> create_interval_metrics
    if skip_node_analysis is False:
        infer_predictions >> node_analysis
