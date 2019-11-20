import os
import os.path as osp
import pandas as pd
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from configparser import ConfigParser
from eai_graph_tools.airflow_tasks.model_tasks import train_graph_model, infer_graph_model
from eai_graph_tools.airflow_tasks.inference_tasks import predict
from eai_graph_tools.airflow_tasks.metrics_tasks import create_interval_metrics
from eai_graph_tools.datasets.intervals import generate_intervals
from eai_graph_tools.airflow_tasks.resources import ResourcePathStatic,\
    ResourcePathById, \
    ResourcePathOutput, \
    HASH_IT, \
    NO_HASH
from eai_graph_tools.airflow_tasks.inference_tasks import nodes_analysis
from tensorboardX import SummaryWriter
from eai_graph_tools.airflow_operators.import_python_persistent_operator import PythonPersistentOperator
from eai_graph_tools.datasets.pytorch_dataset import create_dataset
from eai_graph_tools.airflow_tasks.tabular_tasks import fit_predict_local_outlier_factor, \
    fit_predict_random_forest_classifier, \
    fit_predict_xgboost, \
    fit_predict_random_forest_also

WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']

parser = ConfigParser(converters={'timestamp': lambda t: pd.Timestamp(int(t), unit='s'),
                                  'timedelta': lambda t: pd.Timedelta(int(t), unit='s'),
                                  'nodelist': lambda s: s.splitlines()})


def load_config_file(filename):
    parser.read_file(open(filename))


def setup_experiment_graph_approaches(output_dir, cfg_name):
    Variable.set(cfg_name + 'out_dir', output_dir + cfg_name)
    return "succeeded"


def get_setup_experiment(dag, cfg_name):
    return PythonOperator(
        task_id='setup_experiment',
        force_execution=True,
        python_callable=setup_experiment_graph_approaches,
        op_kwargs={'output_dir': osp.join(WD, parser.get(cfg_name, 'output_path')),
                   'cfg_name': cfg_name},
        dag=dag
    )


def get_trained_model_filename(cfg_name):
    return 'trained_model_' + cfg_name

# ************************************
#
#   GRAPH DATA
#
# ************************************


def get_nodes_of_interest(cfg_name):
    return parser.get(cfg_name, 'nodes_of_interest')


def get_writer(cfg_name):
    print(f"GET WRITER: {osp.join(WD, parser.get(cfg_name, 'output_path'))}")
    return SummaryWriter(osp.join(WD, parser.get(cfg_name, 'output_path'), "runs"), comment=cfg_name)


def get_training_dataset_output_files(cfg_name):
    return generate_intervals(parser.gettimestamp(cfg_name, 'train_start'),
                              parser.gettimestamp(cfg_name, 'train_end'),
                              parser.gettimedelta(cfg_name, 'train_interval_width'),
                              dynamic_path_prefix=[('const', osp.join(WD, parser.get(cfg_name, 'processed_path'))),
                                                   ('var', cfg_name + 'create_training_dataset_hash')])


def get_inference_dataset_output_files(cfg_name):
    return generate_intervals(parser.gettimestamp(cfg_name, 'test_start'),
                              parser.gettimestamp(cfg_name, 'test_end'),
                              parser.gettimedelta(cfg_name, 'test_interval_width'),
                              dynamic_path_prefix=[('const', osp.join(WD, parser.get(cfg_name, 'processed_path'))),
                                                   ('var', cfg_name + 'create_inference_dataset_hash')])


def get_create_training_dataset(dag, cfg_name, force_exec=False):
    """
        create_training_dataset
    """
    task_id = 'create_training_dataset'

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_dataset,
        ppo_kwargs={
            'start': (parser.getint(cfg_name, 'train_start'), HASH_IT),
            'end': (parser.getint(cfg_name, 'train_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'train_interval_width'), HASH_IT),
            'interval_overlap': (parser.getint(cfg_name, 'train_interval_overlap'), HASH_IT),
            'graph_representation': (parser.get(cfg_name, 'graph_representation'), HASH_IT),
            'feature_extractor': (parser.get(cfg_name, 'feature_extractor'), HASH_IT),
        },
        input_files={'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file'))},
        output_files=get_training_dataset_output_files(cfg_name),
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_inference_dataset(dag, cfg_name, force_exec=False):
    """
        create_inference_dataset
    """
    task_id = 'create_inference_dataset'

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_dataset,
        ppo_kwargs={
            'start': (parser.getint(cfg_name, 'test_start'), HASH_IT),
            'end': (parser.getint(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'test_interval_width'), HASH_IT),
            'interval_overlap': (parser.getint(cfg_name, 'test_interval_overlap'), HASH_IT),
            'graph_representation': (parser.get(cfg_name, 'graph_representation'), HASH_IT),
            'feature_extractor': (parser.get(cfg_name, 'feature_extractor'), HASH_IT)
        },
        input_files={'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file'))},
        output_files=get_inference_dataset_output_files(cfg_name),
        dag=dag,
        cfg_name=cfg_name
    )


def get_train_graph_model(dag, cfg_name, force_exec=False):
    """
        train_graph_model
    """
    task_id = 'train_graph_model'

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=train_graph_model,
        ppo_kwargs={
            'start': (parser.gettimestamp(cfg_name, 'train_start'), HASH_IT),
            'end': (parser.gettimestamp(cfg_name, 'train_end'), HASH_IT),
            'interval_width': (parser.gettimedelta(cfg_name, 'train_interval_width'), HASH_IT),
            'hidden_dim': (parser.getint(cfg_name, 'hidden_dim'), HASH_IT),
            'feature_extractor': (parser.get(cfg_name, 'feature_extractor'), HASH_IT),
            'training_epochs': (parser.getint(cfg_name, 'training_epochs'), NO_HASH),
            'predicator_name': (parser.get(cfg_name, 'model_trainer_type'), NO_HASH),
            'tensorboard_writer': (get_writer(cfg_name), NO_HASH),
            'patience_epochs': (parser.getint(cfg_name, 'patience_epochs'), HASH_IT),
            'learning_rate': (parser.getfloat(cfg_name, 'learning_rate'), HASH_IT),
        },
        input_files=get_training_dataset_output_files(cfg_name),
        output_files={
            'trained_model': ResourcePathOutput(cfg_name=cfg_name,
                                                task_id=task_id,
                                                resource_filename=get_trained_model_filename(cfg_name))
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_graph_model_node_embeddings(dag, cfg_name, use_all_nodes=True, force_exec=False):
    """
        infer_graph_model

        use_all_nodes: True, False. When True, nodes_of_interest is set to an empty list and the infer_graph_model
        function will run on all nodes. When False, it gets the list from the ini file.
    """
    task_id = 'create_graph_model_node_embeddings'
    nodes_of_interest = [] if use_all_nodes else parser.getnodelist(cfg_name, get_nodes_of_interest(cfg_name))

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=infer_graph_model,
        ppo_kwargs={
            'start': (parser.gettimestamp(cfg_name, 'test_start'), HASH_IT),
            'end': (parser.gettimestamp(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.gettimedelta(cfg_name, 'test_interval_width'), HASH_IT),
            'predicator_name': (parser.get(cfg_name, 'model_trainer_type'), NO_HASH),
            'hidden_dim': (parser.getint(cfg_name, 'hidden_dim'), HASH_IT),
            'nodes_of_interest': (nodes_of_interest, HASH_IT),
            'tensorboard_writer': (get_writer(cfg_name), NO_HASH),
        },
        input_files={
            **get_inference_dataset_output_files(cfg_name),
            'trained_model': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='train_graph_model',
                                              origin_resource_id='trained_model')
        },
        output_files={
            'node_embeddings': ResourcePathOutput(cfg_name=cfg_name,
                                                  task_id=task_id,
                                                  resource_filename=get_trained_model_filename(cfg_name))
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_predict(dag, cfg_name, use_all_nodes=True, force_exec=False):
    """
        use_all_nodes: True, False. When True, nodes_of_interest is set to an empty list and the predict function
        will run on all nodes. When False, it gets the list from the ini file.
    """
    task_id = 'predict'
    nodes_of_interest = [] if use_all_nodes else parser.getnodelist(cfg_name, get_nodes_of_interest(cfg_name))

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=predict,
        ppo_kwargs={
            'start': (parser.gettimestamp(cfg_name, 'test_start'), HASH_IT),
            'end': (parser.gettimestamp(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.gettimedelta(cfg_name, 'test_interval_width'), HASH_IT),
            'svm_training_technique': (parser.get(cfg_name, 'svm_training_technique'), HASH_IT),
            'nodes_of_interest': (nodes_of_interest, HASH_IT),
            'reference_nodes': (parser.getnodelist(cfg_name, parser.get(cfg_name, 'reference_nodes')), HASH_IT),
            'reference_victim_node': (parser.get(cfg_name, 'reference_victim_node'), HASH_IT),
            'airflow_vars': ({'training_intervals_count': cfg_name + 'training_intervals_count'}, NO_HASH)
        },
        input_files={
            **get_inference_dataset_output_files(cfg_name),
            'node_embeddings': ResourcePathById(cfg_name=cfg_name,
                                                origin_task_id='create_graph_model_node_embeddings',
                                                origin_resource_id='node_embeddings'),
            'trained_model': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='train_graph_model',
                                              origin_resource_id='trained_model')
        },
        output_files={
            'prediction_df': ResourcePathOutput(cfg_name=cfg_name,
                                                task_id=task_id,
                                                resource_filename='df_prediction.h5'),
            'df_metrics': ResourcePathOutput(cfg_name=cfg_name,
                                             task_id=task_id,
                                             resource_filename='df_metrics.h5')
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_interval_metrics(dag, cfg_name, force_exec=False):
    task_id = 'create_interval_metrics'
    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_interval_metrics,
        ppo_kwargs={
            'interval_width': (parser.getint(cfg_name, 'test_interval_width'), HASH_IT),
            'title': (cfg_name, HASH_IT),
        },
        input_files={
            'prediction_df': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='predict',
                                              origin_resource_id='prediction_df')
        },
        output_files={
            'grid_png': ResourcePathOutput(cfg_name=cfg_name,
                                           task_id=task_id,
                                           resource_filename=f"grid_{cfg_name}.png"),
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_node_analysis(dag, cfg_name, force_exec=False):
    task_id = 'node_analysis'

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=nodes_analysis,
        ppo_kwargs={
            'experiment_name': (cfg_name, HASH_IT),
            'start': (parser.gettimestamp(cfg_name, 'test_start'), HASH_IT),
            'end': (parser.gettimestamp(cfg_name, 'test_end'), HASH_IT),
            'nodes_of_interest': (parser.getnodelist(cfg_name, get_nodes_of_interest(cfg_name)), HASH_IT),
            'reference_nodes': (parser.get(cfg_name, parser.get(cfg_name, 'reference_nodes')), HASH_IT),
            'reference_victim_node': (parser.get(cfg_name, 'reference_victim_node'), HASH_IT),
        },
        input_files={
            'df_metrics': ResourcePathById(cfg_name=cfg_name,
                                           origin_task_id='predict',
                                           origin_resource_id='df_metrics')
        },
        output_files={
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
            'df_detailed_classifier_data': ResourcePathOutput(cfg_name=cfg_name,
                                                              task_id=task_id,
                                                              resource_filename='df_detailed_classifier_data.h5'),
            'df_roc_classifier_data': ResourcePathOutput(cfg_name=cfg_name,
                                                         task_id=task_id,
                                                         resource_filename='df_roc_classifier_data.h5')
        },
        dag=dag,
        cfg_name=cfg_name
    )

# ************************************
#
#   TABULAR DATA
#
# ************************************


# ------------------------------------
# Local Outlier Factor (Unsupervised)
# ------------------------------------
def get_fit_predict_local_outlier_factor(dag, cfg_name, force_exec=False, use_smote=False):
    task_id = 'fit_predict_local_outlier_factor'

    cont_variables = parser.get(cfg_name, 'cont_variables').splitlines()
    cat_variables = parser.get(cfg_name, 'cont_variables').splitlines()

    smote_random_state = parser.getint(cfg_name, 'smote_random_state') if use_smote is True else None

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=fit_predict_local_outlier_factor,
        ppo_kwargs={
            'train_start': (parser.getint(cfg_name, 'train_start'), HASH_IT),
            'train_end': (parser.getint(cfg_name, 'train_end'), HASH_IT),
            'test_start': (parser.getint(cfg_name, 'test_start'), HASH_IT),
            'test_end': (parser.getint(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'cont_variables': (cont_variables, HASH_IT),
            'cat_variables': (cat_variables, HASH_IT),
            'n_neighbors': (parser.getint(cfg_name, 'n_neighbors'), HASH_IT),
            'contamination': (parser.getfloat(cfg_name, 'contamination'), HASH_IT),
            'use_smote': (use_smote, HASH_IT),
            'smote_random_state': (smote_random_state, HASH_IT),
        },
        input_files={
            'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file')),
            'features_file': ResourcePathStatic(path=parser.get(cfg_name, 'features_file'))
        },
        output_files={
            'df_all_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                     task_id=task_id,
                                                     resource_filename='df_all_predictions.h5'),
            'df_interval_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                          task_id=task_id,
                                                          resource_filename='df_interval_predictions.h5'),
            'row_based_metrics': ResourcePathOutput(cfg_name=cfg_name,
                                                    task_id=task_id,
                                                    resource_filename=f'row_based_predictions_metrics_{cfg_name}.txt'),
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_interval_metrics_tabular_local_outlier_factor(dag, cfg_name, force_exec=False):
    # Exact same than get_create_interval_metrics,
    # except the input files comes from another task... handle file in-out from higher level?
    task_id = 'create_interval_metrics_tabular_local_outlier_factor'
    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_interval_metrics,
        ppo_kwargs={
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'title': (cfg_name, HASH_IT),
        },
        input_files={
            'prediction_df': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='fit_predict_local_outlier_factor',
                                              origin_resource_id='df_interval_predictions')
        },
        output_files={
            'grid_png': ResourcePathOutput(cfg_name=cfg_name,
                                           task_id=task_id,
                                           resource_filename=f"grid_{cfg_name}.png"),
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
        },
        dag=dag,
        cfg_name=cfg_name
    )


# ------------------------------------
# Random Forest Classifier
# ------------------------------------
def get_fit_predict_random_forest_classifier(dag, cfg_name, force_exec=False):
    task_id = 'fit_predict_random_forest_classifier'

    cont_variables = parser.get(cfg_name, 'cont_variables').splitlines()
    cat_variables = []

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=fit_predict_random_forest_classifier,
        ppo_kwargs={
            'train_start': (parser.getint(cfg_name, 'train_start'), HASH_IT),
            'train_end': (parser.getint(cfg_name, 'train_end'), HASH_IT),
            'test_start': (parser.getint(cfg_name, 'test_start'), HASH_IT),
            'test_end': (parser.getint(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'cont_variables': (cont_variables, HASH_IT),
            'cat_variables': (cat_variables, HASH_IT),
            'n_estimators': (parser.getint(cfg_name, 'n_estimators'), HASH_IT),
            'max_depth': (parser.getint(cfg_name, 'max_depth'), HASH_IT),
            'random_state': (parser.getint(cfg_name, 'random_state'), HASH_IT),
        },
        input_files={
            'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file')),
            'features_file': ResourcePathStatic(path=parser.get(cfg_name, 'features_file'))
        },
        output_files={
            'df_all_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                     task_id=task_id,
                                                     resource_filename='df_all_predictions.h5'),
            'df_interval_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                          task_id=task_id,
                                                          resource_filename='df_interval_predictions.h5'),
            'row_based_metrics': ResourcePathOutput(cfg_name=cfg_name,
                                                    task_id=task_id,
                                                    resource_filename=f'row_based_predictions_metrics_{cfg_name}.txt'),
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_interval_metrics_tabular_random_forest_classifier(dag, cfg_name, force_exec=False):
    # Exact same than get_create_interval_metrics,
    # except the input files comes from another task... handle file in-out from higher level?
    task_id = 'create_interval_metrics_tabular_random_forest_classifier'
    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_interval_metrics,
        ppo_kwargs={
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'title': (cfg_name, HASH_IT),
        },
        input_files={
            'prediction_df': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='fit_predict_random_forest_classifier',
                                              origin_resource_id='df_interval_predictions')
        },
        output_files={
            'grid_png': ResourcePathOutput(cfg_name=cfg_name,
                                           task_id=task_id,
                                           resource_filename=f"grid_{cfg_name}.png"),
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
        },
        dag=dag,
        cfg_name=cfg_name
    )


# ------------------------------------
# Random Forest ALSO
# ------------------------------------
def get_fit_predict_random_forest_also(dag, cfg_name, force_exec=False):
    task_id = 'fit_predict_random_forest_ALSO'

    cont_variables = parser.get(cfg_name, 'cont_variables').splitlines()

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=fit_predict_random_forest_also,
        ppo_kwargs={
            'train_start': (parser.getint(cfg_name, 'train_start'), HASH_IT),
            'train_end': (parser.getint(cfg_name, 'train_end'), HASH_IT),
            'test_start': (parser.getint(cfg_name, 'test_start'), HASH_IT),
            'test_end': (parser.getint(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'cont_variables': (cont_variables, HASH_IT),
            'mean_scaling_threshold': (parser.getfloat(cfg_name, 'mean_scaling_threshold'), HASH_IT),
            'random_forest_max_depth': (parser.getint(cfg_name, 'random_forest_max_depth'), HASH_IT),
            'random_forest_random_state': (parser.getint(cfg_name, 'random_forest_random_state'), HASH_IT),
            'random_forest_n_estimators': (parser.getint(cfg_name, 'random_forest_n_estimators'), HASH_IT),
            'folds': (parser.getint(cfg_name, 'folds'), HASH_IT),
            'samples_training_ratio': (parser.getfloat(cfg_name, 'samples_training_ratio'), HASH_IT),
            'model_type': (parser.get(cfg_name, 'model_type'), HASH_IT),
        },
        input_files={
            'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file')),
            'features_file': ResourcePathStatic(path=parser.get(cfg_name, 'features_file'))
        },
        output_files={
            'df_also_predicted_scores': ResourcePathOutput(cfg_name=cfg_name,
                                                           task_id=task_id,
                                                           resource_filename='df_also_predicted_scores.h5'),
            'df_all_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                     task_id=task_id,
                                                     resource_filename='df_all_predictions.h5'),
            'df_interval_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                          task_id=task_id,
                                                          resource_filename='df_interval_predictions.h5'),
            'row_based_metrics': ResourcePathOutput(cfg_name=cfg_name,
                                                    task_id=task_id,
                                                    resource_filename=f'row_based_predictions_metrics_{cfg_name}.txt'),
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_interval_metrics_tabular_random_forest_also(dag, cfg_name, force_exec=False):
    # Exact same than get_create_interval_metrics,
    # except the input files comes from another task... handle file in-out from higher level?
    task_id = 'create_interval_metrics_tabular_random_forest_ALSO'
    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_interval_metrics,
        ppo_kwargs={
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'title': (cfg_name, HASH_IT),
        },
        input_files={
            'prediction_df': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='fit_predict_random_forest_ALSO',
                                              origin_resource_id='df_interval_predictions')
        },
        output_files={
            'grid_png': ResourcePathOutput(cfg_name=cfg_name,
                                           task_id=task_id,
                                           resource_filename=f"grid_{cfg_name}.png"),
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
        },
        dag=dag,
        cfg_name=cfg_name
    )


# ------------------------------------
# XGBOOST
# ------------------------------------
def get_fit_predict_xgboost(dag, cfg_name, force_exec=False):
    task_id = 'fit_predict_xgboost'

    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=fit_predict_xgboost,
        ppo_kwargs={
            'train_start': (parser.getint(cfg_name, 'train_start'), HASH_IT),
            'train_end': (parser.getint(cfg_name, 'train_end'), HASH_IT),
            'test_start': (parser.getint(cfg_name, 'test_start'), HASH_IT),
            'test_end': (parser.getint(cfg_name, 'test_end'), HASH_IT),
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'importance_type': (parser.get(cfg_name, 'importance_type'), HASH_IT),
        },
        input_files={
            'raw_file': ResourcePathStatic(path=parser.get(cfg_name, 'raw_file')),
            'features_file': ResourcePathStatic(path=parser.get(cfg_name, 'features_file'))
        },
        output_files={
            'df_all_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                     task_id=task_id,
                                                     resource_filename='df_all_predictions.h5'),
            'df_interval_predictions': ResourcePathOutput(cfg_name=cfg_name,
                                                          task_id=task_id,
                                                          resource_filename='df_interval_predictions.h5'),
            'features_importance': ResourcePathOutput(cfg_name=cfg_name,
                                                      task_id=task_id,
                                                      resource_filename='features_importance.txt'),
            'row_based_metrics': ResourcePathOutput(cfg_name=cfg_name,
                                                    task_id=task_id,
                                                    resource_filename=f'row_based_predictions_metrics_{cfg_name}.txt'),
        },
        dag=dag,
        cfg_name=cfg_name
    )


def get_create_interval_metrics_tabular_xgboost(dag, cfg_name, force_exec=False):
    # Exact same than get_create_interval_metrics,
    # except the input files comes from another task... handle file in-out from higher level?
    task_id = 'create_interval_metrics_tabular_xgboost'
    return PythonPersistentOperator(
        task_id=task_id,
        force_execution=force_exec,
        python_callable=create_interval_metrics,
        ppo_kwargs={
            'interval_width': (parser.getint(cfg_name, 'interval_width'), HASH_IT),
            'title': (cfg_name, HASH_IT),
        },
        input_files={
            'prediction_df': ResourcePathById(cfg_name=cfg_name,
                                              origin_task_id='fit_predict_xgboost',
                                              origin_resource_id='df_interval_predictions')
        },
        output_files={
            'grid_png': ResourcePathOutput(cfg_name=cfg_name,
                                           task_id=task_id,
                                           resource_filename=f"grid_{cfg_name}.png"),
            'metrics_summary_file': ResourcePathOutput(cfg_name=cfg_name,
                                                       task_id=task_id,
                                                       resource_filename=f"metrics_summary_file_{cfg_name}.txt"),
        },
        dag=dag,
        cfg_name=cfg_name
    )
