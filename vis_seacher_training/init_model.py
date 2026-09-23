import json
import time
from pathlib import Path

import rich_click as click
import torch
from kostyl.utils import setup_logger
from torch import nn
from torch.utils.data import DataLoader

from vis_seacher_training.configs import TrainingConfig
from vis_seacher_training.data.markup import Items
from vis_seacher_training.datamodule import WineDataModule
from vis_seacher_training.dino_modeling import DinoV3ForWine
from vis_seacher_training.metrics.retrieval import protocol_rows
from vis_seacher_training.metrics.retrieval import recall
from vis_seacher_training.metrics.retrieval import search

logger = setup_logger(fmt="detailed")

CENTERS_NAME = "arcface_centers.pt"
REPORT_NAME = "init_report.json"
FEATURES_NAME = "features.pt"


@torch.no_grad()
def pooled_features(model: DinoV3ForWine, loader: DataLoader, device: torch.device, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """CLS ⊕ GeM бэкбона для всех картинок даталоадера, float32 на CPU, и номера их классов."""
    features, labels = [], []
    started = time.time()
    for n, batch in enumerate(loader):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = model.backbone(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).last_hidden_state
        features.append(model.pool(tokens).float().cpu())
        labels.append(batch["label"])
        if n % 50 == 0:
            logger.info(f"{label}: батч {n + 1}/{len(loader)}, {time.time() - started:.0f} с")
    return torch.cat(features), torch.cat(labels)


def whitening_head(head: nn.Sequential, features: torch.Tensor, embed_dim: int, shrinkage: float, power: float) -> float:
    """Голова BN → Linear → BN становится PCA признаков: стандартизация по признакам, проекция на главные компоненты, выравнивание дисперсий.

    Linear получает стандартизованный вход, поэтому PCA считается по нему же. Второй BN всё равно стандартизует каждую выходную размерность,
    так что степень выравнивания живёт в его γ: power=1 — полное whitening, все размерности равноправны; power=0 — чистая PCA, размерность
    весит корень из своей дисперсии. shrinkage — добавка к собственным числам в долях их среднего, смягчает раздувание шумных компонент.
    Возвращает долю дисперсии, которую держат взятые компоненты.
    """
    bn_in, linear, bn_out = head
    if not (isinstance(bn_in, nn.BatchNorm1d) and isinstance(linear, nn.Linear) and isinstance(bn_out, nn.BatchNorm1d)):
        raise TypeError("ожидается голова BatchNorm1d → Linear → BatchNorm1d")
    x = features.double()
    mean, var = x.mean(0), x.var(0, unbiased=False)
    standardized = (x - mean) / (var + bn_in.eps).sqrt()
    cov = standardized.T @ standardized / (len(x) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # по возрастанию
    top_vals, top_vecs = eigvals.flip(0)[:embed_dim], eigvecs.flip(1)[:, :embed_dim]
    damped = top_vals + shrinkage * eigvals.mean()
    weight = (top_vecs / damped.sqrt()).T  # (embed_dim, in): проекция с единичной дисперсией по каждой компоненте
    projected = standardized @ weight.T
    with torch.no_grad():
        bn_in.running_mean.copy_(mean.float())
        bn_in.running_var.copy_(var.float())
        bn_in.weight.fill_(1.0)
        bn_in.bias.zero_()
        linear.weight.copy_(weight.float())
        linear.bias.zero_()
        bn_out.running_mean.copy_(projected.mean(0).float())
        bn_out.running_var.copy_(projected.var(0, unbiased=False).float())
        bn_out.weight.copy_((damped / damped[0]).pow((1 - power) / 2).float())
        bn_out.bias.zero_()
    return float(top_vals.sum() / eigvals.sum())


@torch.no_grad()
def oneshot_recall(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """recall@k при одном фото класса в галерее, остальные — запросы; как wine/oneshot на валидации."""
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    gallery_rows, rows = protocol_rows(labels, "oneshot")
    _, ids = search(embeddings[rows], embeddings[gallery_rows])
    return recall(labels[gallery_rows][ids] == labels[rows][:, None])


def class_means(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Средний L2-нормированный эмбеддинг каждого класса, снова нормированный."""
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    sums = torch.zeros(num_classes, normalized.shape[1]).index_add_(0, labels, normalized)
    return torch.nn.functional.normalize(sums, dim=1)


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="config.yaml эксперимента: данные, бэкбон, размер входа")
@click.option("-o", "--output", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда сохранить модель save_pretrained, центры и отчёт")
@click.option("--max-images", type=click.IntRange(min=1), default=None, help="Ограничить train этим числом картинок, классы целиком; без значения — data.max_train_images из конфига")
@click.option("--pca-images", type=click.IntRange(min=1), default=20000, show_default=True, help="Сколько картинок train идёт в PCA; средние классов считаются по всем")
@click.option("--val-images", type=click.IntRange(min=1), default=3000, show_default=True, help="Сколько картинок val — на проверку recall до и после")
@click.option("--shrinkage", type=float, default=0.01, show_default=True, help="Добавка к собственным числам в долях среднего: 0 — без сглаживания")
@click.option("--power", type=click.FloatRange(0, 1), default=1.0, show_default=True, help="Степень выравнивания дисперсий у сохраняемой модели: 1 — полное whitening, 0 — чистая PCA; отчёт считается для 1, 0.5 и 0")
@click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--reuse-features", is_flag=True, help="Взять признаки из features.pt в выходной директории, если они там уже есть: так подбирают shrinkage и power без пересчёта бэкбона")
def main(config: Path, output: Path, max_images: int | None, pca_images: int, val_images: int, shrinkage: float, power: float, batch_size: int, reuse_features: bool) -> None:
    """Стартовая модель для обучения: голова из PCA-whitening признаков DINOv3, центры ArcFace из средних эмбеддингов классов.

    Со случайной головой ArcFace долго стоит на плато, а первые шаги портят предобученный бэкбон шумовыми градиентами; отсюда обучение
    начинается с zero-shot качества. В конфиг обучения: model.init_from = <output>, loss.centers_init = <output>/arcface_centers.pt.
    Запуск из корня: python -m vis_seacher_training.init_model.
    """
    cfg = TrainingConfig.from_file(config)
    data_cfg = cfg.data if max_images is None else cfg.data.model_copy(update={"max_train_images": max_images})
    datamodule = WineDataModule(data_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV3ForWine.from_backbone(cfg.model.backbone, embed_dim=cfg.model.embed_dim, gem_p=cfg.model.gem_p, device_map=device, attn_implementation="sdpa").eval()
    model.check_input_size(*data_cfg.input_size)
    output.mkdir(parents=True, exist_ok=True)
    val_items = datamodule.val_items["val"].sample_classes(val_images)
    val_ids = {name: n for n, name in enumerate(sorted(set(val_items.labels)))}
    cache = output / FEATURES_NAME
    if reuse_features and cache.is_file():
        saved = torch.load(cache, weights_only=True)
        if saved["class_names"] != datamodule.class_names or saved["val_names"] != val_items.names:
            raise ValueError(f"{cache} посчитан на другом наборе картинок: пересчитайте без --reuse-features")
        train_features, train_labels, val_features, val_labels = (saved[k] for k in ("train_features", "train_labels", "val_features", "val_labels"))
        logger.info(f"Признаки из {cache}: train {tuple(train_features.shape)}, val {tuple(val_features.shape)}")
    else:

        def loader(items: Items, class_ids: list[int]) -> DataLoader:
            return DataLoader(datamodule._dataset(items, class_ids), batch_size=batch_size, num_workers=data_cfg.num_workers, pin_memory=True)

        label2id = {name: n for n, name in enumerate(datamodule.class_names)}
        train_features, train_labels = pooled_features(model, loader(datamodule.train_items, [label2id[label] for label in datamodule.train_items.labels]), device, "train")
        val_features, val_labels = pooled_features(model, loader(val_items, [val_ids[label] for label in val_items.labels]), device, "val")
        torch.save({"class_names": datamodule.class_names, "val_names": val_items.names, "train_features": train_features, "train_labels": train_labels, "val_features": val_features, "val_labels": val_labels}, cache)

    hidden = train_features.shape[1] // 2
    report: dict[str, object] = {
        "zero_shot_cls": oneshot_recall(val_features[:, :hidden], val_labels),
        "zero_shot_gem": oneshot_recall(val_features[:, hidden:], val_labels),
        "zero_shot_cls_gem": oneshot_recall(val_features, val_labels),
    }
    pca_rows = torch.randperm(len(train_features), generator=torch.Generator().manual_seed(0))[:pca_images]
    model = model.float().cpu()
    for candidate in sorted({1.0, 0.5, 0.0, power}, reverse=True):
        kept = whitening_head(model.head, train_features[pca_rows], cfg.model.embed_dim, shrinkage, candidate)
        with torch.no_grad():
            report[f"head_power_{candidate:g}"] = oneshot_recall(model.head(val_features), val_labels)
    kept = whitening_head(model.head, train_features[pca_rows], cfg.model.embed_dim, shrinkage, power)  # сохраняется запрошенная степень
    with torch.no_grad():
        train_embeddings = model.head(train_features)
    centers = class_means(train_embeddings, train_labels, datamodule.num_classes)
    for name, values in report.items():
        if isinstance(values, dict):
            logger.info(f"val {len(val_items)} картинок, {len(val_ids)} классов, {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))
    logger.info(f"PCA: {len(pca_rows)} векторов, {cfg.model.embed_dim} компонент держат {kept:.1%} дисперсии; сохраняется power={power}, shrinkage={shrinkage}")

    model.save_pretrained(output)
    torch.save({"class_names": datamodule.class_names, "centers": centers}, output / CENTERS_NAME)
    report.update({"backbone": cfg.model.backbone, "input_size": list(data_cfg.input_size), "train_images": len(train_features), "train_classes": datamodule.num_classes, "pca_images": len(pca_rows), "variance_kept": kept, "shrinkage": shrinkage, "power": power})
    (output / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"Модель, {CENTERS_NAME}, {FEATURES_NAME} и {REPORT_NAME} в {output}")


if __name__ == "__main__":
    main()
