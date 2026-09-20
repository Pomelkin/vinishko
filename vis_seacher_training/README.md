# Обучение визуального энкодера

DINOv3 с головой под метрик-лёрнинг (`DinoV3ForWine`: CLS ⊕ GeM по патч-токенам → BN → Linear → BN), лосс sub-center ArcFace
с отступом по размеру класса. Учится на WineSensed, проверяется ретривом. Всё запускается из корня репозитория.

```bash
uv sync --all-groups                        # albumentations, tensorboard, accelerate — в группе vis-searcher-training
hf auth login                               # репозитории DINOv3 на HF закрытые

python -m vis_seacher_training.preview_augs -c vis_seacher_training/experiments/dinov3_vitb16_512/config.yaml -o /tmp/augs.png
python -m vis_seacher_training.run -d vis_seacher_training/experiments/dinov3_vitb16_512
tensorboard --logdir vis_seacher_training/experiments/dinov3_vitb16_512
```

Эксперимент — директория с `config.yaml`. Прогон пишет в её поддиректорию с временем запуска: `config.yaml`, `label2id.json`, `tb_logs/`,
`logs/` (csv), `checkpoints/` и `model/` — лучшие веса в формате `DinoV3ForWine.from_pretrained`, без Lightning и центров ArcFace.
Две видеокарты: `trainer_params.devices: [0, 1]` и `strategy.type: "ddp"`; скорости обучения при этом умножаются на число карт.
Внимание бэкбона выбирает `model.attn_implementation`: `sdpa` по умолчанию, `flash_attention_2` — только с `precision: bf16-mixed`.
У сохранённой модели то же задаётся при загрузке: `DinoV3ForWine.from_pretrained(path, attn_implementation="flash_attention_2", dtype=torch.bfloat16)`.

## Данные

`data.datasets_dir` — раскладка `scripts/prepare_datasets.py` с разметкой `scripts/normalize_dataset.py`; обязаны быть `winesensed`, `off`
и `products10k`, проверяется при чтении конфига. Вход модели — кроп нормализации по сохранённой разметке `normalization.jsonl`: без
аугментаций он совпадает с `render_bottle` пайплайна пиксель в пиксель. Кроп вписывается в `data.input_size` с сохранением пропорций,
поля — нули после нормировки. Обе стороны `input_size` обязаны делиться на `model.patch_size`: проверяется в конфиге, при сборке модели
сверяется с настоящим патчем бэкбона и ещё раз в `forward`.

## Аугментации

`data.augmentations`, порядок применения — в `TrainAugmenter`. Сначала то, что случается при съёмке и вырезке: запас окна над и под
этикеткой (`view.pad_y_range`, пайплайн режет ровно 0.2), шум граней бокса, ошибка выравнивания, перспектива, перекрытия бутылки
и этикетки с порогом видимой доли, порча самой этикетки, блик, свет и баланс белого, шум, смаз, даунскейл и JPEG. Затем то, что делает
пайплайн: заглушение фона по маске, у доли `view.keep_background_prob` примеров пропускается, и ресайз случайной интерполяцией.
Тон почти не трогается: цвет этикетки различает вина одной серии. Посмотреть результат — `preview_augs`; первый батч каждой эпохи
пишется в TensorBoard как `train/augmented_batch`.

## Валидация

Пять даталоадеров: `val` и `distractors` из WineSensed, `negatives` из Products-10K, `catalog` из OFF и `catalog_queries` —
аугментированные виды тех же фото OFF, одинаковые от эпохи к эпохе. В TensorBoard:

- `val/wine/loo`, `val/wine/oneshot` — recall@1/3/5 вина: вся val в галерее либо одно фото на класс, как в каталоге;
- `val/wine/oneshot_noisy` — то же с подмешанной в галерею половиной не-вина; recall@1 отсюда — `val_recall1`, по нему выбирается чекпоинт;
- `val/reject/distractors`, `val/reject/products10k` — ROC AUC и доля принятых запросов val при 1% и 5% ложных срабатываний:
  насколько косинус top-1 отделяет запросы с ответом от вин не из галереи и от не-вина;
- `val/catalog` — recall@k аугментированных запросов по каталогу OFF: замена полевым снимкам, которых для каталога нет;
- `cos_top1/*` — гистограммы косинуса top-1 по группам запросов и общая картинка `cos_top1/groups`.
