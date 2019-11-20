import os

"""
    Environment variable RUN_FROM_REPO_CFG can be used to access the PythonPersistentOperator
    from the repo (as opposed to look for it in the docker image's airflow install)
"""
if 'RUN_FROM_REPO_CFG' not in os.environ:
    # Available after copying the EAI plugin into airflow
    from airflow.operators.python_persistent_operator import PythonPersistentOperator
else:
    from airflow_plugins.plugins.eai_plugin.operators.python_persistent_operator import PythonPersistentOperator

PythonPersistentOperator = PythonPersistentOperator
