import os
import os.path as osp
import pandas as pd
import pathlib
import pytest
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models import TaskInstance, Variable
from configparser import ConfigParser
from datetime import datetime, timedelta
from eai_graph_tools.airflow_tasks.resources import ResourcePathStatic,\
    ResourcePathDynamic, \
    ResourcePathById, \
    HASH_IT, \
    NO_HASH
from eai_graph_tools.airflow_operators.import_python_persistent_operator import PythonPersistentOperator


cfg_name = "unit_test_deep_graph_embeddings_agg_gs_fext_deg_dim10_interval10"
parser = ConfigParser(converters={'timestamp': lambda t: pd.Timestamp(int(t), unit='s'),
                                  'timedelta': lambda t: pd.Timedelta(int(t), unit='s'),
                                  'nodelist': lambda s: s.splitlines()})

WD = "./" if 'WORKDIR' not in os.environ else os.environ['WORKDIR']
parser.read_file(open(WD + "eai_graph_tools/airflow_data/configs/configs_unb2017.ini"))


def overwrite_out_dir_param(path, cfg_name=""):
    Variable.set(cfg_name + 'out_dir', path)


def get_out_dir(cfg_name=""):
    return Variable.get(cfg_name + 'out_dir', default_var='')


default_args = {
    'start_date': datetime.now()
}


@pytest.fixture(scope="session")
def setup_output_path(tmp_path_factory):
    overwrite_out_dir_param(tmp_path_factory.getbasetemp())


def test_resource_missing(setup_output_path):
    out_dir = get_out_dir(cfg_name=cfg_name)
    dag = DAG('test_stat', default_args=default_args, schedule_interval=timedelta(days=1))

    def callable1_dummy_text_read(tp, in_files, out_files, *op_args, **op_kwargs):
        with open(in_files['test_file'].path, 'r') as file:
            assert file.readline() == 'test_data file content', 'Invalid file content!'
            return 'succeeded'
        return 'failed'

    # input file doesn't exist
    input_files = {'test_file': ResourcePathStatic(path=osp.join(out_dir, 'test_data.txt'))}

    read_input_file = PythonPersistentOperator(
        task_id='read_input_file',
        force_execution=True,
        python_callable=callable1_dummy_text_read,
        input_files=input_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti = TaskInstance(task=read_input_file, execution_date=datetime.now())

    with pytest.raises(AirflowException):
        result = read_input_file.execute(ti.get_template_context())
        assert result == 'succeeded'


def test_resource_static_path(setup_output_path):
    out_dir = get_out_dir(cfg_name=cfg_name)
    dag = DAG('test_stat', default_args=default_args, schedule_interval=timedelta(days=1))

    def callable1_dummy_text_read(tp, in_files, out_files, *op_args, **op_kwargs):
        with open(in_files['test_file'].path, 'r') as file:
            assert file.readline() == 'test_data file content', 'Invalid file content!'
            return 'succeeded'
        return 'failed'

    # Creating an input file
    with open(osp.join(out_dir, 'test_data.txt'), 'w') as file:
        file.write("test_data file content")

    input_files = {'test_file': ResourcePathStatic(path=osp.join(out_dir, 'test_data.txt'))}

    read_input_file = PythonPersistentOperator(
        task_id='read_input_file',
        force_execution=True,
        python_callable=callable1_dummy_text_read,
        input_files=input_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti = TaskInstance(task=read_input_file, execution_date=datetime.now())

    result = read_input_file.execute(ti.get_template_context())
    assert result == 'succeeded'


def test_resource_dynamic_path(setup_output_path):
    dag = DAG('test_dyn', default_args=default_args, schedule_interval=timedelta(days=1))

    def callable1_create_file(tp, in_files, out_files, *op_args, **op_kwargs):
        pathlib.Path(osp.dirname(out_files['test_file_dyn_location'].path)).mkdir(parents=True, exist_ok=True)
        with open(out_files['test_file_dyn_location'].path, 'w') as file:
            file.write("testing dynamic paths")
            return 'succeeded'
        return 'failed'

    def callable2_read_file(tp, in_files, out_files, *op_args, **op_kwargs):
        with open(in_files['test_file_dyn_location'].path, 'r') as file:
            assert file.readline() == 'testing dynamic paths', 'Invalid file content!'
            return 'succeeded'
        return 'failed'

    t1_output_files = {
        'test_file_dyn_location': ResourcePathDynamic(path=[('var', cfg_name + 'out_dir'),
                                                            ('var', cfg_name + 'create_file_hash'),
                                                            ('const', 'training'),
                                                            ('const', 'test_data.txt')])}

    create_file = PythonPersistentOperator(
        task_id='create_file',
        force_execution=True,
        python_callable=callable1_create_file,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    t2_input_files = {
        'test_file_dyn_location': ResourcePathDynamic(path=[('var', cfg_name + 'out_dir'),
                                                            ('var', cfg_name + 'create_file_hash'),
                                                            ('const', 'training'),
                                                            ('const', 'test_data.txt')])}

    read_file = PythonPersistentOperator(
        task_id='read_file',
        force_execution=True,
        python_callable=callable2_read_file,
        input_files=t2_input_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file, execution_date=datetime.now())
    ti2 = TaskInstance(task=read_file, execution_date=datetime.now())

    result1 = create_file.execute(ti1.get_template_context())
    result2 = read_file.execute(ti2.get_template_context())

    assert result1 == 'succeeded'
    assert result2 == 'succeeded'


def test_resource_task_indexed_path(setup_output_path):
    dag = DAG('test_dyn', default_args=default_args, schedule_interval=timedelta(days=1))

    # def callable1_create_file(tp, in_files, out_files, *op_args, **op_kwargs):
    def callable1_create_file(log, in_files, out_files, **op_kwargs):
        with open(out_files['test_file_dyn_location'].path, 'w') as file:
            file.write("testing dynamic paths")
            return 'succeeded'
        return 'failed'

    # def callable2_read_file(tp, in_files, out_files, *op_args, **op_kwargs):
    def callable2_read_file(log, in_files, out_files, **op_kwargs):
        with open(in_files['test_file_dyn_location'].path, 'r') as file:
            assert file.readline() == 'testing dynamic paths', 'Invalid file content!'
            return 'succeeded'
        return 'failed'

    create_file = PythonPersistentOperator(
        task_id='create_file',
        force_execution=True,
        python_callable=callable1_create_file,
        output_files={
            'test_file_dyn_location': ResourcePathDynamic(path=[('var', cfg_name + 'out_dir'),
                                                                ('var', cfg_name + 'create_file_hash'),
                                                                ('const', 'training'),
                                                                ('const', 'test_data.txt')])},
        dag=dag,
        cfg_name=cfg_name
    )

    read_file = PythonPersistentOperator(
        task_id='read_file',
        force_execution=True,
        python_callable=callable2_read_file,
        input_files={'test_file_dyn_location': ResourcePathById(cfg_name=cfg_name,
                                                                origin_task_id='create_file',
                                                                origin_resource_id='test_file_dyn_location')},
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file, execution_date=datetime.now())
    ti2 = TaskInstance(task=read_file, execution_date=datetime.now())

    result1 = create_file.execute(ti1.get_template_context())
    result2 = read_file.execute(ti2.get_template_context())

    assert result1 == 'succeeded'
    assert result2 == 'succeeded'


def test_skip_task(setup_output_path):
    out_dir = get_out_dir(cfg_name=cfg_name)
    dag = DAG('test_dyn', default_args=default_args, schedule_interval=timedelta(days=1))

    Variable.set('create_file' + '_hash', 0)

    def callable1_create_file(log, in_files, out_files, **op_kwargs):
        with open(out_files['test_file_dyn_location'].path, 'w') as file:
            file.write("testing dynamic paths")
            return 'succeeded'
        return 'failed'

    # Creating the output file manually
    with open(osp.join(out_dir, 'test_data.txt'), 'w') as file:
        file.write("test_data file content")

    t1_output_files = {'test_file_dyn_location': ResourcePathDynamic(path=[('var', cfg_name + 'out_dir'),
                                                                           ('const', 'test_data.txt')])}

    create_file_forced = PythonPersistentOperator(
        task_id='create_file',
        force_execution=True,
        python_callable=callable1_create_file,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file_forced, execution_date=datetime.now())
    result1 = create_file_forced.execute(ti1.get_template_context())
    assert result1 == 'succeeded'

    create_file_not_forced = PythonPersistentOperator(
        task_id='create_file',
        force_execution=False,
        python_callable=callable1_create_file,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file_not_forced, execution_date=datetime.now())
    result1 = create_file_not_forced.execute(ti1.get_template_context())
    assert result1 == 'skipped'

    # Should run: new params
    some_task_params = {
        'start': (parser.gettimestamp(cfg_name, 'train_start'), HASH_IT),
        'end': (parser.gettimestamp(cfg_name, 'train_end'), NO_HASH),
        'interval_width': (parser.gettimedelta(cfg_name, 'train_interval_width'), HASH_IT),
        'interval_overlap': (parser.gettimedelta(cfg_name, 'train_interval_overlap'), HASH_IT)
    }

    create_file_not_forced = PythonPersistentOperator(
        task_id='create_file',
        force_execution=False,
        python_callable=callable1_create_file,
        ppo_kwargs=some_task_params,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file_not_forced, execution_date=datetime.now())
    result1 = create_file_not_forced.execute(ti1.get_template_context())
    assert result1 == 'succeeded'

    # should skip: no hashed params changed
    some_other_task_params = {
        'start': (parser.gettimestamp(cfg_name, 'train_start'), HASH_IT),
        'end': (parser.gettimestamp(cfg_name, 'train_start'), NO_HASH),
        'interval_width': (parser.gettimedelta(cfg_name, 'train_interval_width'), HASH_IT),
        'interval_overlap': (parser.gettimedelta(cfg_name, 'train_interval_overlap'), HASH_IT)
    }

    create_file_not_forced = PythonPersistentOperator(
        task_id='create_file',
        force_execution=False,
        python_callable=callable1_create_file,
        ppo_kwargs=some_other_task_params,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file_not_forced, execution_date=datetime.now())
    result1 = create_file_not_forced.execute(ti1.get_template_context())
    assert result1 == "skipped"

    # should run: task params changed
    some_other_task_params = {
        'start': (parser.gettimestamp(cfg_name, 'train_end'), HASH_IT),
        'end': (parser.gettimestamp(cfg_name, 'train_end'), NO_HASH),
        'interval_width': (parser.gettimedelta(cfg_name, 'train_interval_width'), HASH_IT),
        'interval_overlap': (parser.gettimedelta(cfg_name, 'train_interval_overlap'), HASH_IT)
    }

    create_file_not_forced = PythonPersistentOperator(
        task_id='create_file',
        force_execution=False,
        python_callable=callable1_create_file,
        ppo_kwargs=some_other_task_params,
        output_files=t1_output_files,
        dag=dag,
        cfg_name=cfg_name
    )

    ti1 = TaskInstance(task=create_file_not_forced, execution_date=datetime.now())
    result1 = create_file_not_forced.execute(ti1.get_template_context())
    assert result1 == 'succeeded'
