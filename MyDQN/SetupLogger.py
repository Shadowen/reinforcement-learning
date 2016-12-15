import inspect
from enum import Enum, unique
import json


@unique
class LogType(Enum):
    environment = 1
    optimizer = 2
    estimator = 3
    replay_memory = 4
    algorithm = 5


class SetupLogger():
    def __init__(self):
        self.logs = {}
        self._log_type_to_logging_function = {}

        self.create_custom_logger(LogType.environment)
        self.create_custom_logger(LogType.optimizer)
        self.create_custom_logger(LogType.estimator, functions=['model_builder'],
            exceptions=['sess', 'env', 'optimizer'])
        self.create_custom_logger(LogType.replay_memory)
        self.create_custom_logger(LogType.algorithm, functions=['get_td_target'],
            exceptions=['env', 'q_estimator', 'target_estimator', 'replay_memory'])

    def create_custom_logger(self, log_type, functions=[], exceptions=[]):
        def logger(function, *args, **kwargs):
            # Log the function's args
            self.logs[str(log_type)] = {'type':function.__name__, 'args':args}
            # Log selected functions as strings
            self.logs[str(log_type)].update({f:inspect.getsource(kwargs[f]) for f in functions})
            # Log the kwargs, except the ones that are handled separately
            self.logs[str(log_type)].update({k:v for k, v in kwargs.items() if k not in functions + exceptions})

        self._log_type_to_logging_function[log_type] = logger

    def logged_call(self, function, log_type=None):
        def wrapper(*args, **kwargs):
            self._log_type_to_logging_function[log_type](function, *args, **kwargs)
            return function(*args, **kwargs)

        return wrapper

    def dump_to_file(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.logs, file)


if __name__ == '__main__':
    # Check all log types have a logger
    for log_type in LogType:
        if log_type not in SetupLogger()._log_type_to_logging_function:
            print('Warning: missing logger for log type {}'.format(str(log_type)))
