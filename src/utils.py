# import
from ruamel.yaml import safe_load
from os.path import isfile
import random
from torchaudio.functional import lowpass_biquad, highpass_biquad
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchaudio

# def


def load_yaml(filepath):
    with open(file=filepath, mode='r') as f:
        config = safe_load(f)
    return config


def get_sox_effect_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        effect_dict = {}
        effect_config = load_yaml(filepath=filepath)
        for stage in effect_config.keys():
            effect_dict[stage] = []
            if type(effect_config[stage]) != dict:
                effect_dict[stage] = None
                continue
            for effect_type, values in effect_config[stage].items():
                effect_dict[stage].append([effect_type, '{}'.format(
                    random.uniform(min(values), max(values)))])
        return effect_dict
    else:
        assert False, 'please check the sox effect config path: {}'.format(
            filepath)


def digital_filter(waveform, filter_type, sample_rate, cutoff_freq):
    if filter_type == 'bandpass':
        waveform = lowpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=max(cutoff_freq), Q=1)
        waveform = highpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=min(cutoff_freq), Q=1)
    elif filter_type == 'lowpass':
        waveform = lowpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=max(cutoff_freq), Q=1)
    elif filter_type == 'highpass':
        waveform = highpass_biquad(
            waveform=waveform, sample_rate=sample_rate, cutoff_freq=min(cutoff_freq), Q=1)
    return waveform


def pad_waveform(waveform, max_waveform_length):
    diff = max_waveform_length-len(waveform)
    pad = (0, int(diff))
    waveform = F.pad(input=waveform, pad=pad)
    return waveform


def get_transform_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = {}
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
            for transform_type in transform_config[stage].keys():
                temp = []
                for name, value in transform_config[stage][transform_type].items():
                    if transform_type not in ['audio', 'vision']:
                        assert False, 'please check the transform config.'
                    module_name = 'torchaudio.transforms' if transform_type == 'audio' else 'torchvision.transforms'
                    if value is None:
                        temp.append(eval('{}.{}()'.format(module_name, name)))
                    else:
                        if type(value) is dict:
                            value = ('{},'*len(value)).format(*
                                                              ['{}={}'.format(a, b) for a, b in value.items()])
                        temp.append(
                            eval('{}.{}({})'.format(module_name, name, value)))
                if transform_type == 'audio':
                    transform_dict[stage][transform_type] = nn.Sequential(
                        *temp)
                elif transform_type == 'vision':
                    transform_dict[stage][transform_type] = torchvision.transforms.Compose(
                        temp)
        return transform_dict
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)
