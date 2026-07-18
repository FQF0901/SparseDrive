# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import sys
import os

print(sys.executable, os.path.abspath(__file__))
# import init_paths # for conda pkgs submitting method
import argparse
import copy
import mmcv
import time
import torch
# PyTorch 2.6+ changed torch.load weights_only default to True,
# but mmcv 1.x doesn't pass it — override default for legacy .pth files.
_orig_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_torch_load
# PyTorch 2.6+ changed _get_stream to require torch.device not int
# mmcv passes int device_ids through Scatter.forward -> _get_stream
from torch.nn.parallel._functions import _get_stream as _orig_get_stream
def _patched_get_stream(device):
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    return _orig_get_stream(device)
import torch.nn.parallel._functions as _torch_parallel_fns
_torch_parallel_fns._get_stream = _patched_get_stream
# PyTorch 2.6+ removed DDP._use_replicated_tensor_module attribute
# that mmcv MMDistributedDataParallel._run_ddp_forward relies on.
from mmcv.parallel import MMDistributedDataParallel
_orig_ddp_forward = MMDistributedDataParallel._run_ddp_forward
def _patched_ddp_forward(self, *inputs, **kwargs):
    if hasattr(self, '_use_replicated_tensor_module'):
        module_to_run = self._replicated_tensor_module if \
            self._use_replicated_tensor_module else self.module
    else:
        module_to_run = self.module
    if self.device_ids:
        inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
        return module_to_run(*inputs[0], **kwargs[0])
    else:
        return module_to_run(*inputs, **kwargs)
MMDistributedDataParallel._run_ddp_forward = _patched_ddp_forward
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from torch import distributed as dist
from datetime import timedelta

import cv2

cv2.setNumThreads(8)


def parse_args():
    # torchrun passes --local-rank (hyphen) but argparse expects --local_rank (underscore)
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--local-rank"):
            sys.argv[i] = "--local_rank" + arg[len("--local-rank"):]
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument("--gpus-per-machine", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    elif args.launcher == "mpi_nccl":
        distributed = True

        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        print(
            "MPI local_rank=%d, world_size=%d"
            % (mpi_local_rank, mpi_world_size)
        )

        # num_gpus = torch.cuda.device_count()
        device_ids_on_machines = list(range(args.gpus_per_machine))
        str_ids = list(map(str, device_ids_on_machines))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=mpi_world_size,
            rank=mpi_local_rank,
            timeout=timedelta(seconds=3600),
        )

        cfg.gpu_ids = range(mpi_world_size)
        print("cfg.gpu_ids:", cfg.gpu_ids)
    else:
        distributed = True
        init_dist(
            args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params
        )
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(
        "Environment info:\n" + dash_line + env_info + "\n" + dash_line
    )
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()
    logger.info(f"Model:\n{model}")

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    if hasattr(cfg, "plugin"):
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )
    else:
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method(
        "fork"
    )  # use fork workers_per_gpu can be > 1
    main()
