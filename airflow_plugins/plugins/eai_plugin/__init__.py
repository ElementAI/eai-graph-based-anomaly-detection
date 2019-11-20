# This is the class you derive to create a plugin
from airflow.plugins_manager import AirflowPlugin
from .operators.python_persistent_operator import PythonPersistentOperator


# Defining the plugin class
class python_persistent_operator(AirflowPlugin):
    name = "python_persistent_operator"
    operators = [PythonPersistentOperator]


HASH_IT = True
NO_HASH = False
