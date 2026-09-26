# Обучение визуального энкодера

DINOv3 с головой под метрик-лёрнинг (`DinoV3ForWine`: CLS ⊕ GeM по патч-токенам → BN → Linear → BN), лосс sub-center ArcFace
с отступом по размеру класса. Учится на WineSensed, проверяется ретривом. Всё запускается из корня репозитория.

```bash
uv sync --all-groups                        # albumentations, tensorboard, accelerate — в группе vis-searcher-training
hf auth login                               # репозитории DINOv3 на HF закрытые

python -m vis_seacher_training.preview_augs -c vis_seacher_training/experiments/dinov3_vitb16_512/config.yaml -o /tmp/augs.png
python -m vis_seacher_training.augs_fixture save -c vis_seacher_training/experiments/dinov3_vitb16_512/config.yaml -o /tmp/augs_ref.npz  # эталон перед правкой аугментаций, потом check
python -m vis_seacher_training.run -d vis_seacher_training/experiments/dinov3_vitb16_512
tensorboard --logdir vis_seacher_training/experiments/dinov3_vitb16_512
```

Эксперимент — директория с `config.yaml`. Прогон пишет в её поддиректорию с временем запуска: `config.yaml`, `label2id.json`, `tb_logs/`,
`logs/` (csv), `checkpoints/` и `model/` — лучший чекпоинт, поднятый через `DinoV3ForWine.from_lightning_checkpoint`, в трёх видах:
веса `save_pretrained` без Lightning и центров ArcFace; `model.onnx` одним файлом, батч и стороны картинки динамические (кратные патчу),
на входе float32 RGB NCHW со значениями 0…255 — нормировка ImageNet вшита в граф, на выходе L2-нормированные эмбеддинги; `preprocess.json`
с описанием входа. Экспорт сверяется с PyTorch через onnxruntime на двух формах. Отдельно: `python -m vis_seacher_training.export -c <ckpt> -o <dir>`.
Рядом пишется `model.bf16.onnx` — тот же граф в bfloat16 для TensorRT (`--no-bf16` отключает): в TensorRT 11 точность задаётся типами
графа, а onnxruntime bf16 не исполняет; сверяется через TensorRT, если он установлен (группа `flash-inference`).
Замер обученной модели тем же протоколом, что TULIP и EVIE, для сравнения столбец в столбец: `python -m scripts.bench_dino.run --onnx <dir>/model.onnx ...`
(`--backend tensorrt --onnx <dir>/model.bf16.onnx` — через TensorRT, engine кэшируется рядом с ONNX), отчёт в `reports/dino/`.
Две видеокарты: `trainer_params.devices: [0, 1]` и `strategy.type: "ddp"`; скорости обучения при этом умножаются на число карт.
Внимание бэкбона выбирает `model.attn_implementation`: `sdpa` по умолчанию, `flash_attention_2` — только с `precision: bf16-mixed`.
У сохранённой модели то же задаётся при загрузке: `DinoV3ForWine.from_pretrained(path, attn_implementation="flash_attention_2", dtype=torch.bfloat16)`.

## Данные

`data.datasets_dir` — раскладка `scripts/prepare_datasets.py` с разметкой `scripts/normalize_dataset.py`; обязаны быть `winesensed`, `off`
и `products10k`, проверяется при чтении конфига. Вход модели — кроп нормализации по сохранённой разметке `normalization.jsonl`: без
аугментаций он совпадает с `render_bottle` пайплайна пиксель в пиксель. Кроп вписывается в `data.input_size` с сохранением пропорций,
поля добавляются в uint8 цветом заливки фона из normalize.toml до нормировки, как сделает пайплайн перед энкодером. Обе стороны `input_size` обязаны делиться на `model.patch_size`: проверяется в конфиге, при сборке модели
сверяется с настоящим патчем бэкбона и ещё раз в `forward`.

## Стартовая модель

Со случайной головой ArcFace долго стоит на плато, а первые шаги портят предобученный бэкбон шумовыми градиентами: zero-shot DINOv3 ViT-L
даёт recall@1 ≈ 0.95 на val, а после ста шагов со случайной головой — 0.003. Поэтому обучение начинается с подготовленной модели:

```bash
python -m vis_seacher_training.init_model -c <config.yaml> -o weights/init/<имя>   # ~2 ч на 3080 для всего train
```

Скрипт прогоняет бэкбон по train, делает голову BN → Linear → BN PCA-whitening'ом признаков (`--power` — степень выравнивания дисперсий,
`--shrinkage` — сглаживание собственных чисел), считает центры ArcFace как средние эмбеддинги классов и пишет отчёт с recall на val для
zero-shot и для головы. Признаки кэшируются в `features.pt`: `--reuse-features` подбирает параметры головы без пересчёта бэкбона.
В конфиг обучения: `model.init_from: <директория>` и `loss.centers_init: <директория>/arcface_centers.pt`; классы без записи в файле
получают случайные центры.
Продолжить обучение с более широкой головой: `python -m vis_seacher_training.widen_head -m <run>/model -f weights/init/<имя>/features.pt -o weights/init/<имя>_cont --embed-dim 1024` —
бэкбон и GeM как есть, γ и β обоих BN сброшены, обученные строки Linear остаются, недостающие берутся из PCA остатка признаков поверх них, центры пересчитываются;
пример конфига продолжения — `experiments/dinov3_vitl16_1024_cont`. `model.drop_path_rate` и `model.attention_dropout` — регуляризация бэкбона. `hyperparams.backbone_freeze_ratio` держит бэкбон на нулевом lr первые столько шагов, потом у него свой
разогрев и косинус; голова и центры учатся с первого шага.

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
- `cos_top1/*` — гистограммы косинуса top-1 по группам запросов и общая картинка `cos_top1/groups`;
- `misses/oneshot_noisy` — `eval.log_misses` случайных промахов главного протокола картинкой: запрос, фото его класса из галереи
  с рангом, на котором оно нашлось, и top-5 найденного с косинусами; рамка зелёная у фото класса запроса, красная у чужого вина, жёлтая у не-вина.
