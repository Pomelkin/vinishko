# Установка Docker Desktop на Windows

На этой машине есть `winget` и WSL 2.7.14; команда `docker` пока не установлена. Для Docker Desktop используйте WSL 2 backend.

1. Проверьте в **Диспетчере задач → Производительность → ЦП**, что «Виртуализация» включена. Если выключена, включите Intel VT-x или AMD-V в BIOS/UEFI.
2. Откройте PowerShell **от имени администратора** и обновите WSL:

   ```powershell
   wsl --update
   wsl --version
   ```

   Если WSL отсутствует на другой машине, выполните `wsl --install`, затем перезагрузите Windows.
3. Установите Docker Desktop:

   ```powershell
   winget install --id Docker.DockerDesktop --exact --source winget
   ```

4. Запустите **Docker Desktop** из меню «Пуск». При первом запуске примите условия, выберите **Use WSL 2 based engine** и дождитесь состояния *Engine running*. Если установщик попросит выйти из системы или перезагрузиться, сделайте это.
5. Откройте новое окно PowerShell и проверьте установку:

   ```powershell
   docker version
   docker compose version
   docker run --rm hello-world
   ```

   Если `docker` не распознаётся после установки, терминал использует старый `PATH`. Для текущего окна выполните:

   ```powershell
   $env:Path = "C:\Program Files\Docker\Docker\resources\bin;$env:Path"
   docker version
   ```

   Затем перезапустите терминал или редактор, из которого он открыт.

## Запуск сомелье

Из корня репозитория создайте файл с реальным ключом OpenRouter:

```powershell
Copy-Item secrets/openrouter_api_key.txt.example secrets/openrouter_api_key.txt
notepad secrets/openrouter_api_key.txt
docker compose up --build -d
```

Замените текст примера в файле ключом. Файл исключён из Git и образа. Сервис доступен на `http://127.0.0.1:8000`.

Проверка через HTTP-клиент (с реальным запросом к OpenRouter):

```powershell
uv sync
uv run python scripts/smoke_api.py
```

Клиент создаёт UUIDv7, получает первый ответ, отправляет вопрос, читает историю и удаляет тестовую сессию. Для ручного диалога есть [notebook](../notebooks/sommelier_api.ipynb). Если нужен другой адрес сервиса, передайте `--base-url` скрипту или установите `SOMMELIER_BASE_URL` для notebook.
