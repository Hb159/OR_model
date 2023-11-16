# -*- coding: utf-8 -*-
# @author: HB
# @email: haobin@buaa.edu.cn
# @date: 2022/10/22
import os
from dataclasses import dataclass

# 标准化路径
project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class DataPath:
    data_root_path: str = os.path.join(project_dir, 'data')

    raw_data_path: str = os.path.join(data_root_path, 'data_raw')
    intermediate_data_path: str = os.path.join(data_root_path, 'data_intermediate')
    clean_data_path: str = os.path.join(data_root_path, 'data_clean')


@dataclass(frozen=True)
class ResultPath:
    result_path: str = os.path.join(project_dir, 'result')
    solver_result_root_path: str = os.path.join(project_dir, result_path, 'solver_result')

    log_dir: str = os.path.join(project_dir, solver_result_root_path, 'log')
    iis_dir: str = os.path.join(project_dir, solver_result_root_path, 'iis')
    mps_dir: str = os.path.join(project_dir, solver_result_root_path, 'mps')
    sol_dir: str = os.path.join(project_dir, solver_result_root_path, 'sol')
    lp_dir: str = os.path.join(project_dir, solver_result_root_path, 'lp')

    data_print_path: str = os.path.join(project_dir, result_path, 'print_result')

    figure_path: str = os.path.join(project_dir, result_path, 'figure_result')


@dataclass(frozen=True)
class DataProcessPath:
    data_process_root_path: str = os.path.join(project_dir, 'data_process')
    init_file: str = os.path.join(data_process_root_path, '__init__.py')

    data_preprocess_dir: str = os.path.join(data_process_root_path, 'data_preprocess')
    data_process_dir: str = os.path.join(data_process_root_path, 'data_process')
    data_postprocess_dir: str = os.path.join(data_process_root_path, 'data_postprocess')

    preprocess_init_file: str = os.path.join(data_process_root_path, '__init__.py')
    process_init_file: str = os.path.join(data_process_root_path, '__init__.py')
    post_process_init_file: str = os.path.join(data_process_root_path, '__init__.py')


@dataclass(frozen=True)
class ConfigPath:
    config_root_path: str = os.path.join(project_dir, 'config')
    init_file: str = os.path.join(config_root_path, '__init__.py')


@dataclass(frozen=True)
class ControllerPath:
    controller_root_path: str = os.path.join(project_dir, 'controller')
    init_file: str = os.path.join(controller_root_path, '__init__.py')


@dataclass(frozen=True)
class UtilsPath:
    utils_root_path: str = os.path.join(project_dir, 'utils')
    init_file: str = os.path.join(utils_root_path, '__init__.py')


@dataclass(frozen=True)
class ModelPath:
    model_root_path: str = os.path.join(project_dir, 'model')
    init_file: str = os.path.join(model_root_path, '__init__.py')


def create_file_dictionary():
    class_list = [
        ConfigPath(),
        ControllerPath(),
        DataPath(),
        DataProcessPath(),
        ModelPath(),
        ResultPath(),
        UtilsPath(),
    ]
    for classpath in class_list:
        for name, filepath in classpath.__dict__.items():
            if not os.path.exists(filepath):
                if name != 'init_file':
                    os.makedirs(filepath)
                    print(f'create filepath: {filepath}')
                else:
                    with open(filepath, 'wb') as f:
                        pass
                    print(f'create file: {filepath}')


if __name__ == '__main__':
    create_file_dictionary()

