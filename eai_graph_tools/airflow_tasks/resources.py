from airflow.exceptions import AirflowException
from airflow.models import Variable
import os.path as osp
import json

HASH_IT = True
NO_HASH = False


class ResourcePath:
    pass


class ResourcePathStatic(ResourcePath):

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        if self._path is not None:
            return self._path
        else:
            raise AirflowException("Static path is undefined")


class ResourcePathDynamic(ResourcePath):

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        path_list = []
        for item in self._path:
            if item[0] == 'const':
                path_list.append(item[1])
            elif item[0] == 'var':
                path_list.append(Variable.get(item[1], default_var=''))
            else:
                AirflowException('Dynamic_paths dict must either have "var" of "const" as keys')
        return osp.join(*path_list)


class ResourcePathById(ResourcePath):

    def __init__(self,
                 cfg_name,
                 origin_task_id,
                 origin_resource_id):

        self.cfg_name = cfg_name
        self.origin_task_id = origin_task_id
        self.origin_resource_id = origin_resource_id

    @property
    def path(self):
        output_files = json.loads(Variable.get(f'{self.cfg_name + self.origin_task_id}_output_files',
                                               default_var=''))
        output_files_dict = {}
        for item in output_files:
            output_files_dict[item[0]] = item[1]
        return output_files_dict[self.origin_resource_id]


class ResourcePathOutput(ResourcePath):

    def __init__(self,
                 cfg_name,
                 task_id,
                 resource_filename):
        self.cfg_name = cfg_name
        self.task_id = task_id
        self.resource_filename = resource_filename

    @property
    def path(self):
        path_list = [Variable.get(self.cfg_name + 'out_dir', default_var=''),
                     Variable.get(self.cfg_name + self.task_id + '_hash', default_var=''),
                     self.resource_filename]
        return osp.join(*path_list)
