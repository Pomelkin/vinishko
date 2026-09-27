import os
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import cast

import lightning as L
import rich_click as click
import torch
from kostyl.ml.configs import DDPStrategyConfig
from kostyl.ml.configs import SingleDeviceStrategyConfig
from kostyl.ml.configs import SupportedStrategies
from kostyl.ml.configs.structs.training_settings import FSDP1StrategyConfig
from kostyl.ml.configs.structs.training_settings import FSDP2StrategyConfig
from kostyl.ml.integrations.lightning.callbacks import setup_checkpoint_callback
from kostyl.ml.integrations.lightning.callbacks import setup_early_stopping_callback
from kostyl.ml.integrations.lightning.loggers.tb_logger import setup_tb_logger
from kostyl.utils import setup_logger
from lightning import Callback
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.strategies import Strategy

from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import load_config
from vis_seacher_training.configs import TrainingConfig
from vis_seacher_training.datamodule import WineDataModule
from vis_seacher_training.export import export_model
from vis_seacher_training.training_module import WineTrainingModule


logger = setup_logger(fmt="detailed")

torch.set_float32_matmul_precision("high")

RUN_NAME_ENV = "VIS_SEARCHER_RUN_NAME"


def setup_strategy(
    accelerator: str, strategy_settings: SupportedStrategies, devices: list[int] | int
) -> Strategy:
    """Стратегия Lightning по конфигу: одно устройство либо DDP."""
    device_ids = devices if isinstance(devices, list) else list(range(devices))
    if not device_ids:
        raise ValueError("Device list cannot be empty.")
    parallel_devices = [
        torch.device(accelerator, index)
        if accelerator != "cpu"
        else torch.device("cpu")
        for index in device_ids
    ]
    match strategy_settings:
        case DDPStrategyConfig():
            if len(device_ids) < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            return DDPStrategy(
                accelerator=accelerator,
                parallel_devices=parallel_devices,
                find_unused_parameters=strategy_settings.find_unused_parameters,
            )
        case SingleDeviceStrategyConfig():
            if len(device_ids) != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")
            return SingleDeviceStrategy(
                device=parallel_devices[0], accelerator=accelerator
            )
        case FSDP1StrategyConfig():
            if len(device_ids) < 2:
                raise ValueError("FSDP1 strategy requires at least two devices.")
            return FSDPStrategy(
                accelerator=accelerator,
                parallel_devices=parallel_devices,
            )
        case FSDP2StrategyConfig():
            if len(device_ids) < 2:
                raise ValueError("FSDP2 strategy requires at least two devices.")
            return ModelParallelStrategy(
                data_parallel_size=len(device_ids),
                tensor_parallel_size=1,
                save_distributed_checkpoint=False,
            )
        case _:
            raise ValueError(
                f"Unsupported strategy: {strategy_settings.__class__.__name__}"
            )


def resolve_run_name(run_name: str | None) -> str:
    """Имя прогона, общее для всех рангов DDP: дочерние процессы перезапускают скрипт и сами взяли бы уже другое время."""
    return os.environ.setdefault(
        RUN_NAME_ENV,
        run_name or f"{datetime.now(tz=UTC).astimezone():%Y-%m-%d_%H-%M-%S}",
    )


@click.command()
@click.option(
    "--experiment-dir",
    "-d",
    type=click.Path(
        exists=True, file_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Директория эксперимента: из неё читается config.yaml, в поддиректорию прогона пишутся логи TensorBoard, чекпоинты и итоговая модель.",
)
@click.option(
    "--run-name",
    default=None,
    help="Имя поддиректории прогона; по умолчанию время запуска.",
)
def run(experiment_dir: Path, run_name: str | None) -> None:
    """Обучение DinoV3ForWine. Запуск из корня: python -m vis_seacher_training.run -d vis_seacher_training/experiments/<имя>.

    Логи: tensorboard --logdir <experiment-dir>.
    """
    config = TrainingConfig.from_file(experiment_dir / "config.yaml")
    L.seed_everything(config.seed, workers=True)
    work_dir = experiment_dir / resolve_run_name(run_name)
    config.dump_to_file(work_dir / "config.yaml")

    datamodule = WineDataModule(config.data, workdir=work_dir)
    training_module = WineTrainingModule(config=config)

    params = config.trainer_params
    strategy = setup_strategy(
        accelerator=params.accelerator,
        strategy_settings=params.strategy,
        devices=params.devices,
    )
    callbacks: list[Callback] = []
    if config.early_stopping is not None:
        callbacks.append(setup_early_stopping_callback(config.early_stopping))
    callbacks.append(
        setup_checkpoint_callback(
            dirpath=work_dir / "checkpoints", ckpt_cfg=config.checkpointing
        )
    )
    callbacks.append(
        LearningRateMonitor(
            logging_interval="step", log_weight_decay=True, log_momentum=False
        )
    )

    trainer = Trainer(
        max_epochs=params.max_epochs,
        accelerator="auto",
        devices=params.devices,
        strategy=strategy,
        precision=params.precision,
        accumulate_grad_batches=params.accumulate_grad_batches,
        gradient_clip_val=config.hyperparams.grad_clip_val,
        val_check_interval=params.val_check_interval,
        callbacks=callbacks,
        log_every_n_steps=params.log_every_n_steps,
        limit_train_batches=params.limit_train_batches,
        limit_val_batches=params.limit_val_batches,
        logger=[
            setup_tb_logger(work_dir),
            CSVLogger(save_dir=work_dir, name="logs", version="version_0"),
        ],
    )
    trainer.fit(training_module, datamodule=datamodule)
    model = training_module.best_model()
    if model is not None and trainer.is_global_zero:
        model.float()
        fill = tuple(
            int(v)
            for v in load_config(
                config.data.normalize_config or NORM_DIR / "normalize.toml", []
            )["background"]["color"]
        )
        export_model(
            model,
            work_dir / "model",
            config.data.input_size,
            cast(tuple[int, int, int], fill),
            config.data.img_mean,
            config.data.img_std,
        )


if __name__ == "__main__":
    run()
