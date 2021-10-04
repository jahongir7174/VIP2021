import argparse
import copy
import logging
import os
import random
import time

import apex
import mmcv
import numpy
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_optimizer, build_runner, get_dist_info, init_dist)
from mmcv.utils import build_from_cfg, get_git_hash, collect_env, get_logger

import mmdet
from mmdet import __version__
from mmdet.datasets import (build_dataloader, build_dataset, replace_pipeline)
from mmdet.models import build_detector
from tools.test import inference


class DistributedEvalHook(mmcv.runner.DistEvalHook):

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        results = inference(runner.model, self.dataloader)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


def set_random_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_detector(model, dataset, cfg, timestamp, meta):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [build_dataloader(ds,
                                     cfg.data.samples_per_gpu,
                                     cfg.data.workers_per_gpu,
                                     len(cfg.gpu_ids),
                                     dist=True,
                                     seed=cfg.seed) for ds in dataset]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    model, optimizer = apex.amp.initialize(model.cuda(), optimizer)
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True

    model = MMDistributedDataParallel(model.cuda(),
                                      device_ids=[torch.cuda.current_device()],
                                      broadcast_buffers=False)

    # build runner
    runner = build_runner(cfg.runner,
                          default_args=dict(model=model,
                                            optimizer=optimizer,
                                            work_dir=cfg.work_dir,
                                            logger=logger,
                                            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks, support batch_size > 1 in validation
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_pipeline(cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(val_dataset,
                                      samples_per_gpu=val_samples_per_gpu,
                                      workers_per_gpu=cfg.data.workers_per_gpu,
                                      dist=True,
                                      shuffle=False)
    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_hook = DistributedEvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), 'Each item in custom_hooks expects dict type, but got ' \
                                               f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = os.path.join('./weights',
                                os.path.splitext(os.path.basename(args.config))[0])
    cfg.gpu_ids = range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    init_dist('pytorch', **cfg.dist_params)
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info_dict['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: True')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    logger.info(f'Set random seed to o, deterministic: True')
    set_random_seed()
    cfg.seed = 0
    meta['seed'] = 0
    meta['exp_name'] = os.path.basename(args.config)

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7],
                                          CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, timestamp, meta)


if __name__ == '__main__':
    main()
