# import
import argparse
import torch
from src.utils import load_yaml
from os.path import abspath, isfile
import numpy as np
from timm import list_models

# class


class ProjectParameters:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # base
        self._parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune', 'evaluate'], required=True,
                                  help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self._parser.add_argument(
            '--data_path', type=str, required=True, help='the data path.')
        self._parser.add_argument(
            '--random_seed', type=self._str_to_int, default=0, help='the random seed.')
        self._parser.add_argument(
            '--save_path', type=str, default='save/', help='the path which stores the checkpoint of PyTorch lightning.')
        self._parser.add_argument('--no_cuda', action='store_true', default=False,
                                  help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self._parser.add_argument('--gpus', type=self._str_to_int_list, default=-1,
                                  help='number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node. if give -1 will use all available GPUs.')
        self._parser.add_argument(
            '--parameters_config_path', type=str, default=None, help='the parameters config path.')

        # data preparation
        self._parser.add_argument('--sample_rate', type=self._str_to_int, required=True,
                                  help='sample rate of audio signal.')
        self._parser.add_argument('--max_waveform_length', type=self._str_to_int, required=True,
                                  help='the maximum length of the waveform (seconds).')
        self._parser.add_argument('--filter_type', type=self._str_to_str, default='bandpass', choices=[
                                  'bandpass', 'lowpass', 'highpass', None], help='the filter type provided bandpass, lowpass, highpass, and None.')
        self._parser.add_argument('--cutoff_freq', type=self._str_to_int_list,
                                  default=[1000, 2000], help='filter cutoff frequency.')
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')
        self._parser.add_argument('--classes', type=self._str_to_str_list, required=True,
                                  help='the classes of data.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        self._parser.add_argument(
            '--no_balance', action='store_true', default=False, help='whether to balance the data.')
        self._parser.add_argument('--transform_config_path', type=self._str_to_str,
                                  default='config/transform.yaml', help='the transform config path.')
        self._parser.add_argument('--sox_effect_config_path', type=self._str_to_str,
                                  default='config/sox_effect.yaml', help='the sox effect config path.')
        self._parser.add_argument(
            '--cells', type=int, default=6, help='how many cells use for detection.')
        self._parser.add_argument(
            '--boxes', type=int, default=2, help='how many boxes in each cell.')

        # model
        self._parser.add_argument('--backbone_model', type=str, required=True,
                                  help='if you want to use a self-defined model, give the path of the self-defined model. otherwise, the provided backbone model is as a followed list. {}'.format(list_models()))
        self._parser.add_argument('--checkpoint_path', type=str, default=None,
                                  help='the path of the pre-trained model checkpoint.')
        self._parser.add_argument('--optimizer_config_path', type=str,
                                  default='config/optimizer.yaml', help='the optimizer config path.')
        self._parser.add_argument('--confidence', type=float, default=0.5,
                                  help='the confidence used to check the keyword whether exist.')
        self._parser.add_argument('--gap', type=float, default=0.01,
                                  help='the gap to separate two identicle words.')

        # debug
        self._parser.add_argument(
            '--max_files', type=self._str_to_int, default=None, help='the maximum number of files for loading files.')

    def _str_to_int(self, s):
        return None if s == 'None' or s == 'none' else int(s)

    def _str_to_int_list(self, s):
        return [int(v) for v in s.split(',') if len(v) > 0]

    def _str_to_str(self, s):
        return None if s == 'None' or s == 'none' else s

    def _str_to_str_list(self, s):
        if '.txt' in s:
            return list(np.loadtxt(abspath(s), str))
        else:
            return [str(v) for v in s.split(',') if len(v) > 0]

    def _get_new_dict(self, old_dict, yaml_dict):
        for k in yaml_dict.keys():
            del old_dict[k]
        return {**old_dict, **yaml_dict}

    def parse(self):
        project_parameters = self._parser.parse_args()
        if project_parameters.parameters_config_path is not None:
            project_parameters = argparse.Namespace(**self._get_new_dict(old_dict=vars(
                project_parameters), yaml_dict=load_yaml(filepath=abspath(project_parameters.parameters_config_path))))
        else:
            del project_parameters.parameters_config_path

        # base
        project_parameters.data_path = abspath(
            path=project_parameters.data_path)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda
        project_parameters.gpus = project_parameters.gpus if project_parameters.use_cuda else 0

        # data preparation
        project_parameters.classes = sorted(project_parameters.classes)
        project_parameters.max_waveform_length *= project_parameters.sample_rate
        project_parameters.class_to_idx = {
            c: idx for idx, c in enumerate(project_parameters.classes)}
        project_parameters.num_classes = len(project_parameters.classes)
        assert not any((np.array(project_parameters.cutoff_freq)/project_parameters.sample_rate)
                       > 1), "please check the cutoff_freq whether it satisfies Nyquist's theorem."
        project_parameters.use_balance = not project_parameters.no_balance
        if project_parameters.transform_config_path is not None:
            project_parameters.transform_config_path = abspath(
                project_parameters.transform_config_path)
        if project_parameters.sox_effect_config_path is not None:
            project_parameters.sox_effect_config_path = abspath(
                project_parameters.sox_effect_config_path)

        # model
        project_parameters.optimizer_config_path = abspath(
            project_parameters.optimizer_config_path)
        if isfile(project_parameters.backbone_model):
            project_parameters.backbone_model = abspath(
                project_parameters.backbone_model)
        if project_parameters.checkpoint_path is not None and isfile(project_parameters.checkpoint_path):
            project_parameters.checkpoint_path = abspath(
                project_parameters.checkpoint_path)
        project_parameters.output_features = project_parameters.cells * \
            (project_parameters.boxes*3+project_parameters.num_classes)

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
