import collections
import copy
import warnings

import numpy
import torch
from mmcv.runner.hooks import HOOKS, Hook

from mmdet.models.rpn import RPNHead


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, numpy.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, collections.abc.Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def replace_pipeline(pipelines):
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_pipeline(pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn('"ImageToTensor" pipeline is replaced by '
                          '"DefaultFormatBundle" for batch inference. It is '
                          'recommended to manually replace it in the test '
                          'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


@HOOKS.register_module()
class NumClassCheckHook(Hook):

    @staticmethod
    def _check_head(runner):
        model = runner.model
        dataset = runner.data_loader.dataset
        if dataset.CLASSES is None:
            runner.logger.warning(f'Please set `CLASSES` '
                                  f'in the {dataset.__class__.__name__} and'
                                  f'check if it is consistent with the `num_classes` '
                                  f'of head')
        else:
            assert type(dataset.CLASSES) is not str, (f'`CLASSES` in {dataset.__class__.__name__}'
                                                      f'should be a tuple of str.'
                                                      f'Add comma if number of classes is 1 as '
                                                      f'CLASSES = ({dataset.CLASSES},)')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not isinstance(module, RPNHead):
                    assert module.num_classes == len(dataset.CLASSES), (f'The `num_classes` ({module.num_classes}) in '
                                                                        f'{module.__class__.__name__} of '
                                                                        f'{model.__class__.__name__} does not matches '
                                                                        f'the length of `CLASSES` '
                                                                        f'{len(dataset.CLASSES)}) in '
                                                                        f'{dataset.__class__.__name__}')

    def before_train_epoch(self, runner):
        self._check_head(runner)

    def before_val_epoch(self, runner):
        self._check_head(runner)
