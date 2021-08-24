# import
from src.project_parameters import ProjectParameters
from torchvision.datasets import DatasetFolder
import torchaudio
from src.utils import get_sox_effect_from_file, digital_filter, pad_waveform, get_transform_from_file
import numpy as np
from os.path import join
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

# class


class SpeechYoloDataSet(DatasetFolder):
    def __init__(self, root: str, transform, project_parameters, stage):
        super().__init__(root, extensions=('.wav'), loader=None)
        self.transform = transform
        self.project_parameters = project_parameters
        self.stage = stage

    def __getitem__(self, index: int):
        # get features
        filepath, _ = self.samples[index]
        data, sample_rate = torchaudio.load(filepath=filepath)
        assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate and input sample_rate. the sample_rate: {}, the input sample_rate: {}'.format(
            sample_rate, self.project_parameters.sample_rate)
        if self.project_parameters.sox_effect_config_path is not None:
            effects = get_sox_effect_from_file(
                filepath=self.project_parameters.sox_effect_config_path)[self.stage]
            if effects is not None:
                data, _ = torchaudio.sox_effects.apply_effects_tensor(
                    tensor=data, sample_rate=sample_rate, effects=effects)
        assert sample_rate == self.project_parameters.sample_rate, 'please check the sample_rate and input sample_rate. the sample_rate: {}, the input sample_rate: {}'.format(
            sample_rate, self.project_parameters.sample_rate)
        if self.project_parameters.filter_type is not None:
            data = digital_filter(waveform=data, filter_type=self.project_parameters.filter_type,
                                  sample_rate=sample_rate, cutoff_freq=self.project_parameters.cutoff_freq)
        if len(data[0]) < self.project_parameters.max_waveform_length:
            data = pad_waveform(
                waveform=data[0], max_waveform_length=self.project_parameters.max_waveform_length)[None]
        else:
            data = data[:, :self.project_parameters.max_waveform_length]
        if self.transform is not None:
            data = self.transform['audio'](data)
            if 'vision' in self.transform:
                data = self.transform['vision'](data)

        # get target and kwspotting target
        _, _, feature_len = data.shape
        divide = sample_rate/feature_len
        width_cell = feature_len / self.project_parameters.cells
        yolo_data = []
        with open(filepath[:-3]+'txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\t', ' ').split(' ')
                start = np.floor(float(line[0])/divide)
                end = np.floor(float(line[1])/divide)
                object_width = end-start
                center_x = start + object_width / 2.0
                cell_index = int(center_x / width_cell)
                object_norm_x = float(center_x) / width_cell - cell_index
                object_norm_w = object_width/feature_len
                object_class = self.project_parameters.class_to_idx[line[2]]
                yolo_data.append([cell_index, object_norm_x,
                                  object_norm_w, object_class])
        kwspotting_target = torch.ones(
            [len(self.project_parameters.classes)]) * -1
        target = torch.zeros([self.project_parameters.cells, (self.project_parameters.boxes *
                                                              3+len(self.project_parameters.classes)+1)], dtype=torch.float32)
        for (cell_index, object_norm_x, object_norm_w, object_class) in yolo_data:
            target[cell_index, self.project_parameters.boxes*3 + object_class] = 1
            target[cell_index, -1] = 1
            for box in range(self.project_parameters.boxes):
                target[cell_index, box*3 + 2] = 1
                target[cell_index, box * 3] = object_norm_x
                target[cell_index, box * 3 + 1] = object_norm_w
            kwspotting_target[object_class] = 1
        return data, target, kwspotting_target


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        self.dataset = {}
        for stage in ['train', 'val', 'test']:
            self.dataset[stage] = SpeechYoloDataSet(root=join(self.project_parameters.data_path, stage),
                                                    stage=stage, transform=self.transform_dict[stage], project_parameters=self.project_parameters)
            # modify the maximum number of files
            if self.project_parameters.max_files is not None:
                lengths = (self.project_parameters.max_files, len(
                    self.dataset[stage])-self.project_parameters.max_files)
                self.dataset[stage] = random_split(
                    dataset=self.dataset[stage], lengths=lengths)[0]
        if self.project_parameters.max_files is not None:
            assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.class_to_idx, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                self.dataset['train'].dataset.class_to_idx, self.project_parameters.class_to_idx)
        else:
            assert self.dataset['train'].class_to_idx == self.project_parameters.class_to_idx, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                self.dataset['train'].class_to_idx, self.project_parameters.class_to_idx)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()

    #
    for x, y, z in data_loaders['train']:
        break
    print(x.shape, y.shape, z.shape)
