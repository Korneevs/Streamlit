# Streamlit Multi-App Portal (Auto-discovery)

**URL портала:** https://msk.avito-streamlit.ru:8501  
Портал автоматически подхватывает любые приложения из `/srv/portal/apps/*/app.yaml` и встраивает их в iframe: `https://msk.avito-streamlit.ru:<port>/`.

## Структура
```
/srv/portal/
  ├─ portal_streamlit/main.py
  ├─ apps/
  │   ├─ promiser/app.yaml
  │   └─ matrexa/app.yaml
  ├─ scripts/ensure_apps.sh
  └─ systemd/
      ├─ streamlit-portal.service
      └─ streamlit@.service
```

## Запуск портала
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PORTAL_CANONICAL_ORIGIN="https://msk.avito-streamlit.ru:8501"
export APPS_ROOT="/srv/portal/apps"
streamlit run portal_streamlit/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

## Добавление приложений
Каждая аппка: `/srv/portal/apps/<slug>/app.yaml` + `src/entry.py`.

Пример `app.yaml`:
```yaml
slug: promiser
title: "pROMIser"
description: "Предсказание эффективности флайта (DTB)"
icon: "📈"
port: 8511
entry: "streamlit_app.py"
```

## Автоподхват (опционально, через systemd/cron)
```bash
sudo cp scripts/ensure_apps.sh /usr/local/bin/
# в crontab root:
*/5 * * * * /usr/local/bin/ensure_apps.sh
```
