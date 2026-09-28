# Веб-интерфейс

`static/` содержит `index.html`, `style.css` и `app.js`. HTML/CSS/JS без сборки и внешних зависимостей.
Основной FastAPI отдаёт страницу по `/`, ресурсы — по `/static/`.

Из корня проекта в одном Python-окружении запустите основной API и микросервисы в трёх отдельных терминалах.
Ключ, proxy и путь к истории настройте по [инструкции запуска](../README.md#запуск-всего-приложения).

```powershell
python -m vinishko.app
python -m uvicorn vinishko.app.sommelier.main:app --env-file .env --host 127.0.0.1 --port 8805 --workers 1
python -m uvicorn vinishko.app.whatis.main:app --env-file .env --host 127.0.0.1 --port 8810 --workers 1
```

Откройте `http://127.0.0.1:8000/`. Доступны загрузка фото, выбор маски бутылки, карточки и кандидаты,
whatis, чат с подсказками, восстановление истории и удаление диалога, ввод карточки вручную.

Проверка логики без браузера и OpenRouter:

```powershell
node vinishko/app/frontend/test_frontend.cjs
```
