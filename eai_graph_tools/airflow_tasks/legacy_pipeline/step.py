import os
import os.path as osp
import pickle
from abc import ABC, ABCMeta, abstractmethod
from hashlib import sha512
from typeguard import check_type
from typing import Callable, Mapping, Tuple, Type, TypeVar  # type: ignore
from warnings import warn

T = TypeVar("T")


# Derived from https://github.com/rusty1s/pytorch_geometric/blob/84e26aac42f681a88c98d444f7e5c83733f7ed7a/torch_geometric/data/dataset.py#L19  # noqa: E501
class PersistentStep(ABC):
    def __init__(self, root_dir, input_dir, step_name, *args):
        super().__init__()
        self.root = osp.expanduser(osp.normpath(root_dir))
        self.inp_dir = input_dir
        self.out_dir = osp.join(self.root, f'{step_name}/output')
        self.step_name = step_name
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.pickle_path = self.hash_pickle_path(root_dir, input_dir, step_name, *args)

    @abstractmethod
    def input_file_names(self):
        pass

    def input_paths(self):
        return [osp.join(self.inp_dir, f) for f in self.input_file_names]

    @abstractmethod
    def output_file_names(self):
        pass

    def output_paths(self):
        return [osp.join(self.out_dir, f) for f in self.output_file_names]

    @abstractmethod
    def _execute(self):
        pass

    def hash_pickle_path(self, *args):
        h = sha512()
        for arg in args:
            h.update(str(arg).encode("utf-8"))
        return osp.join(self.root, '%s.pk' % h.hexdigest())

    def run(self):
        out_files = self.output_paths() + [self.pickle_path]
        # If all of the output exists and is newer than the input, skip
        if all([osp.exists(f) for f in out_files]) and \
           min([osp.getmtime(f) for f in out_files]) >= \
           max([osp.getmtime(f) for f in self.input_paths()]):
            print(f"Skipping {self.step_name} for cached output...")
            return False

        if not all([osp.exists(f) for f in self.input_paths()]):
            missing = [f for f in self.input_paths() if not osp.exists(f)]
            raise RuntimeError("All of the required files do not exist in %s, missing %s" % (self.__class__.__name__,
                                                                                             str(missing)))

        print(f"Executing {self.step_name}...")
        mem_out = self._execute()
        pickle.dump(mem_out, open(self.pickle_path, 'wb'))
        return True

    def get_output(self):
        return pickle.load(open(self.pickle_path, 'rb'))


class StepChecker(ABCMeta):
    """
    This is a metaclass

    For more information see https://docs.python.org/3/reference/datamodel.html?highlight=metaclass#metaclasses

    By extending ABCMeta, all classes which use StepChecker as a metaclass get all of the behavior they would from
    extending ABC (which just adds ABCMeta as a metaclass under the hood)

    The __call__ method is called when a class is instantiated, so we can check at runtime whether
    Input and Output have been implemented. This matches the behavior of ABCMeta, so all of the @abstractmethod's will
    be checked at the same time

    The __init__ method is called when a class is defined, so we can check before any code is run whether a particular
    subclass has implemented "run" and potentially avoided type checking
    """
    MEMBER_CLASSES = ["Input", "Output"]
    PROTECTED_METHODS = ["run"]

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        for member in StepChecker.MEMBER_CLASSES:
            if not (hasattr(cls, member) and (
                    # _GenericAlias comes from typing and is the base of its type annotations
                    # isinstance(getattr(cls, member), type) or isinstance(getattr(cls, member), _GenericAlias))):
                    isinstance(getattr(cls, member), type))):
                raise TypeError("Subclass of Step must implement a class " + member)
        # Pass on instantiation to ABCMeta
        return super(StepChecker, cls).__call__(*args, **kwargs)

    def __init__(self, cls: Type[T], namespace: Tuple, members: Mapping):
        for method in StepChecker.PROTECTED_METHODS:
            # The second condition adds an exception for the original class
            if method in members and members["__qualname__"] != "Step":
                warn("Subclasses of Step should not override " + method,
                     category=RuntimeWarning)


# N.B. Adding an Input and Output to this class will nullify the effects of StepChecker
class Step(metaclass=StepChecker):

    def run(self, x):
        # Typecheck the input and output by deferring to the typing module's method
        # This works for both Python's type and typing's _GenericAlias
        check_type("Input", x, self.Input)
        out = self._run(x)
        check_type("Output", out, self.Output)
        return out

    @abstractmethod
    def _run(self, x):
        pass

# Some utilities to help with the new structure. Some of these may turn out to be more useful than others


def create_step_class(inp: Type, out: Type, f: Callable) -> Type[Step]:
    class NewStep(Step):
        Input = inp
        Output = out

        def _run(self, i):
            return f(i)

    return NewStep


def create_step(inp: Type, out: Type, f: Callable) -> Step:
    return create_step_class(inp, out, f)()


# Allow the decorator pattern to build the class structure around a function
def pipeline_step(inp: Type, out: Type) -> Callable:
    def decorator(f: Callable) -> Callable:
        # Ignore typing since Mypy doesn't support dynamic base classes
        # https://github.com/python/mypy/issues/2477
        class DecoratorStep(create_step_class(inp, out, f)):  # type: ignore

            # Implement __call__ to retain expected function behavior
            def __call__(self, *args):
                return f(*args)

        return DecoratorStep()
    return decorator
