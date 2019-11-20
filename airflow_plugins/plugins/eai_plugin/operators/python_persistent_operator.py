from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.utils.operator_helpers import context_to_airflow_vars
from airflow.models import Variable
from hashlib import md5
from typing import Optional, Dict, Iterable, Callable, Mapping, Any, Tuple, AnyStr
import os
import pickle
import os.path as osp
import pathlib
import json

HASH_IT = True
NO_HASH = False


# To be used as airflow.operators.test_plugin.PluginOperator
class PythonPersistentOperator(BaseOperator):
    template_fields = ('templates_dict', 'op_args', 'op_kwargs')
    ui_color = '#ffefeb'

    # since we won't mutate the arguments, we should just do the shallow copy
    # there are some cases we can't deepcopy the objects(e.g protobuf).
    shallow_copy_attrs = ('python_callable', 'op_kwargs',)

    @apply_defaults
    def __init__(
        self,
        python_callable: Callable,
        op_args: Optional[Iterable] = None,
        ppo_kwargs: Optional[Mapping[str, Tuple[Any, bool]]] = None,
        provide_context: bool = False,
        force_execution: bool = False,
        cfg_name: Optional[AnyStr] = None,
        task_params: Optional[Mapping[str, Tuple[Any, bool]]] = None,
        input_files: Optional[Dict] = None,
        output_files: Optional[Dict] = None,
        templates_dict: Optional[Dict] = None,
        templates_exts: Optional[Iterable[str]] = None,
        op_kwargs: Optional[Dict] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not callable(python_callable):
            raise AirflowException('`python_callable` param must be callable')
        if cfg_name is None:
            raise AirflowException('`cfg_name` param is required (for paths variable naming)')

        self.task_params = task_params

        self.cfg_name = cfg_name
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs
        self.ppo_kwargs_with_hashing_info = ppo_kwargs if ppo_kwargs is not None else {}
        self.force_execution = force_execution
        self.params = kwargs['params']
        self.python_callable = python_callable
        self.provide_context = provide_context
        self.input_files = input_files
        self.output_files = output_files
        self._provide_op_kwargs = None

        self.templates_dict = templates_dict
        if templates_exts:
            self.template_ext = templates_exts

    @property
    def ppo_kwargs_without_hashing_info(self):
        return dict((k, v) for k, (v, _) in self.ppo_kwargs_with_hashing_info.items())

    def params_hash(self):
        params_subset = {}

        for k, (value, decision_hash) in self.ppo_kwargs_with_hashing_info.items():
            if decision_hash == HASH_IT:
                params_subset[k] = value
        return md5(pickle.dumps(params_subset)).hexdigest()

    def get_variables_prefix(self):
        return self.cfg_name + self.task_id

    def skip_task_check(self, params_hash):
        skip_task = True

        self.log.info(f"params_hash: {params_hash}\nVariable.get(self.get_var_prefix() + '_hash'): "
                      f"{Variable.get(self.get_variables_prefix() + '_hash', default_var='')}")

        if Variable.get(self.get_variables_prefix() + '_hash', default_var='') == params_hash:
            self.log.info(f"\tskip_task_check....hash hasn't changed since last run")
            new_hash = False
        else:
            Variable.set(self.get_variables_prefix() + '_hash', params_hash)
            self.log.info(f"\tskip_task_check....hash changed since last run")
            new_hash = True

        if self.force_execution or new_hash is True:
            self.log.info(f"\tskip_task_check....won't skip task (force exec = {self.force_execution}, "
                          f"new_hash={new_hash})")
            skip_task = False
        elif self.output_files is None:
            self.log.info(f"\tskip_task_check....won't skip task (self.output_files is None)")
            skip_task = False
        elif all([osp.exists(f.path) for _, f in self.output_files.items()]):
            if self.input_files is None:
                skip_task = True
                self.log.info(f"\tskip_task_check....will skip task (no input file, all output files present)")
            else:
                min_val = min([osp.getmtime(f.path) for _, f in self.output_files.items()])
                max_val = max([osp.getmtime(f.path) for _, f in self.input_files.items()])
                if min_val >= max_val:
                    self.log.info(f"\tskip_task_check....will skip task (outputs files are more recent "
                                  f"than inputs files)")
                    skip_task = True
                else:
                    self.log.info(f"\tskip_task_check....won't skip task (outputs files are less recent than "
                                  f"inputs files)")
                    skip_task = False
        return skip_task

    def saving_output_filenames(self):
        if self.output_files is not None:
            output_files_var = []
            for k, v in self.output_files.items():
                # making sure directory exists
                pathlib.Path(osp.dirname(v.path)).mkdir(parents=True, exist_ok=True)
                output_files_var.append((k, v.path))
            # Saving the output files list as Airflow variable
            Variable.set(f"{self.get_variables_prefix()}_output_files", json.dumps(output_files_var))

    def execute(self, context):
        self.log.info(f'Executing {self.task_id} (Forcing execution: {self.force_execution})')

        if self.input_files is not None:
            self.log.info(f'Input files:')
            input_files = [f.path for _, f in self.input_files.items()]
            self.log.info(f'{str(input_files)}')

            if not all([osp.exists(f.path) for _, f in self.input_files.items()]):
                missing = [f.path for _, f in self.input_files.items() if not osp.exists(f.path)]
                raise AirflowException("All required files don't exist in %s, missing %s" % (self.__class__.__name__,
                                                                                             str(missing)))

        if self.output_files is not None:
            self.log.info(f'Output files:')
            output_files = [f.path for _, f in self.output_files.items()]
            self.log.info(f'{str(output_files)}')

        # Export context to make it available for callables to use.
        airflow_context_vars = context_to_airflow_vars(context, in_env_var_format=True)
        self.log.info("Exporting the following env vars:\n%s",
                      '\n'.join(["{}={}".format(k, v)
                                 for k, v in airflow_context_vars.items()]))

        os.environ.update(airflow_context_vars)

        if self.provide_context:
            context.update(self.ppo_kwargs_without_hashing_info)
            context['templates_dict'] = self.templates_dict
            self.op_kwargs = context

        if self.skip_task_check(self.params_hash()):
            self.log.info(f'Skipping Task')

            return_value = "skipped"
            self.log.info("Skipping execution, output already exists. Returning: %s", return_value)
        else:
            self.saving_output_filenames()
            return_value = self.execute_callable()
            self.log.info("Done. Returned value was: %s", return_value)
        return return_value

    def execute_callable(self):
        return self.python_callable(self.log,
                                    self.input_files,
                                    self.output_files,
                                    **self.ppo_kwargs_without_hashing_info)
