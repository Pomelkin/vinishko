import copy
import json
from pathlib import Path

import rich_click as click
import torch
from kostyl.utils import setup_logger
from torch import nn

from vis_seacher_training.dino_modeling import DinoV3ForWine
from vis_seacher_training.init_model import CENTERS_NAME
from vis_seacher_training.init_model import class_means
from vis_seacher_training.init_model import oneshot_recall


logger = setup_logger(fmt="detailed")

REPORT_NAME = "widen_report.json"


def standardized(features: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Вход Linear головы: признаки после первого BN в eval с γ=1, β=0."""
    return (
        (features.double() - bn.running_mean.double())  # ty: ignore[unresolved-attribute]
        / (bn.running_var.double() + bn.eps).sqrt()  # ty: ignore[unresolved-attribute]
    ).float()


def residual_pca(
    inputs: torch.Tensor, kept: torch.Tensor, count: int, shrinkage: float
) -> tuple[torch.Tensor, float]:
    """count главных компонент остатка признаков после проекции на span уже обученных строк Linear, с выравниванием дисперсий.

    Обученные строки уже описывают какое-то подпространство; новые направления берутся из того, что в него не попало, иначе половина
    новой головы дублировала бы старую. Возвращает строки (count, in) и долю дисперсии остатка, которую они держат.
    """
    x = inputs.double()
    basis, _ = torch.linalg.qr(
        kept.double().T
    )  # ортонормированный базис span обученных строк, (in, kept)
    residual = x - (x @ basis) @ basis.T
    cov = residual.T @ residual / (len(x) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    top_vals, top_vecs = eigvals.flip(0)[:count], eigvecs.flip(1)[:, :count]
    damped = top_vals + shrinkage * eigvals.clamp(min=0).mean()
    return (top_vecs / damped.sqrt()).T.float(), float(
        top_vals.sum() / eigvals.clamp(min=0).sum()
    )


@click.command()
@click.option(
    "-m",
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="save_pretrained обученной модели, например <run>/model",
)
@click.option(
    "-f",
    "--features",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="features.pt из init_model.py: CLS ⊕ GeM train и val с метками; признаки бэкбона той же архитектуры, пусть и до дообучения",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда сохранить расширенную модель, центры ArcFace и отчёт",
)
@click.option(
    "--embed-dim",
    type=click.IntRange(min=1),
    default=1024,
    show_default=True,
    help="Новая размерность эмбеддинга; обученные строки Linear остаются, недостающие берутся из PCA остатка",
)
@click.option(
    "--pca-images", type=click.IntRange(min=1), default=20000, show_default=True
)
@click.option("--shrinkage", type=float, default=0.01, show_default=True)
def main(
    model_dir: Path,
    features: Path,
    output: Path,
    embed_dim: int,
    pca_images: int,
    shrinkage: float,
) -> None:
    """Расширить голову обученной модели до embed_dim для продолжения обучения: бэкбон и GeM как есть, γ и β обоих BN сброшены,
    первые строки Linear — обученные, остальные — PCA остатка признаков поверх них; центры ArcFace пересчитываются под новую голову.

    Признаки берутся из кэша init_model.py, то есть от бэкбона до дообучения: он сдвинулся на проценты, для стартовых направлений этого
    хватает, а статистики BN выправятся за первые шаги. Запуск из корня: python -m vis_seacher_training.widen_head.
    """
    trained = DinoV3ForWine.from_pretrained(
        model_dir, attn_implementation="sdpa", dtype=torch.float32
    ).eval()
    old_dim = trained.config.embed_dim
    if embed_dim < old_dim:
        raise click.BadParameter(
            f"embed_dim {embed_dim} меньше обученного {old_dim}: сужать голову этот скрипт не умеет",
            param_hint="--embed-dim",
        )
    cache = torch.load(features, weights_only=True)
    train_features, train_labels, val_features, val_labels = (
        cache[k]
        for k in ("train_features", "train_labels", "val_features", "val_labels")
    )
    if train_features.shape[1] != trained.head[0].num_features:
        raise click.BadParameter(
            f"признаки шириной {train_features.shape[1]}, а голова ждёт {trained.head[0].num_features}",
            param_hint="--features",
        )

    config = copy.deepcopy(trained.config)
    config.embed_dim = embed_dim
    model = DinoV3ForWine(config).eval()
    state = {k: v for k, v in trained.state_dict().items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("head.") for k in missing):
        raise RuntimeError(
            f"веса не сошлись: лишние {unexpected}, не хватает {[k for k in missing if not k.startswith('head.')]}"
        )
    bn_in, linear, bn_out = model.head
    old_in, old_linear, _ = trained.head
    if not (
        isinstance(bn_in, nn.BatchNorm1d)
        and isinstance(linear, nn.Linear)
        and isinstance(bn_out, nn.BatchNorm1d)
    ):
        raise TypeError("ожидается голова BatchNorm1d → Linear → BatchNorm1d")

    rows = torch.randperm(
        len(train_features), generator=torch.Generator().manual_seed(0)
    )[:pca_images]
    with torch.no_grad():
        bn_in.running_mean.copy_(old_in.running_mean)
        bn_in.running_var.copy_(old_in.running_var)
        bn_in.weight.fill_(1.0)
        bn_in.bias.zero_()
        inputs = standardized(train_features, bn_in)
        extra, kept = residual_pca(
            inputs[rows], old_linear.weight.detach(), embed_dim - old_dim, shrinkage
        )
        linear.weight.copy_(torch.cat([old_linear.weight.detach(), extra]))
        linear.bias.copy_(
            torch.cat([old_linear.bias.detach(), torch.zeros(embed_dim - old_dim)])
        )
        projected = linear(inputs)
        bn_out.running_mean.copy_(projected.mean(0))
        bn_out.running_var.copy_(projected.var(0, unbiased=False))
        bn_out.weight.fill_(1.0)
        bn_out.bias.zero_()
        embeddings = model.head(train_features)
        report = {
            "val_oneshot_trained_head": oneshot_recall(
                trained.head(val_features), val_labels
            ),
            "val_oneshot_widened_head": oneshot_recall(
                model.head(val_features), val_labels
            ),
        }
    centers = class_means(embeddings, train_labels, len(cache["class_names"]))
    for name, values in report.items():
        logger.info(
            f"val {len(val_features)} картинок по кэшу признаков, {name}: "
            + ", ".join(f"{k} {v:.3f}" for k, v in values.items())  # ty: ignore[unresolved-attribute]
        )
    logger.info(
        f"голова {old_dim} → {embed_dim}: {embed_dim - old_dim} новых строк из PCA остатка держат {kept:.1%} его дисперсии; γ и β обоих BN сброшены"
    )

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    torch.save(
        {"class_names": cache["class_names"], "centers": centers}, output / CENTERS_NAME
    )
    report.update(
        {
            "source": str(model_dir),
            "features": str(features),
            "old_embed_dim": old_dim,
            "embed_dim": embed_dim,
            "residual_variance_kept": kept,
            "pca_images": len(rows),
            "shrinkage": shrinkage,
        }
    )
    (output / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(f"Модель, {CENTERS_NAME} и {REPORT_NAME} в {output}")


if __name__ == "__main__":
    main()
