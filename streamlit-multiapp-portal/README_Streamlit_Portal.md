# Streamlit Multi-App Portal

Единый портал, который автоматически подхватывает любое число внутренних Streamlit-приложений из `apps/*/app.yaml`
и открывает их во фрейме на одном URL портала.

---

## 🚀 Быстрый старт (локально)

1. Клонируйте репозиторий:
   ```bash
   git clone git@github.com:Korneevs/Streamlit.git
   cd Streamlit
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустите портал:
   ```bash
   export APPS_ROOT="./apps"
   export PORTAL_CANONICAL_ORIGIN="http://localhost:8501"
   streamlit run portal_streamlit/main.py --server.port=8501 --server.address=0.0.0.0
   ```

4. В другом терминале запустите одну из аппок, например `promiser`:
   ```bash
   streamlit run apps/promiser/src/streamlit_pROMIser.py --server.port=8511
   ```

5. Перейдите в браузере на [http://localhost:8501](http://localhost:8501)

---

## 🧩 Формат `app.yaml`

Каждая аппка должна содержать в корне файл `app.yaml`:

```yaml
slug: promiser
title: pROMIser
description: Оптимизация рекламных кампаний
icon: 📈
port: 8511
entry: src/streamlit_pROMIser.py
```

---

## 🖥️ Деплой на сервер

1. Скопируйте проект в `/srv/portal/`
2. Установите зависимости (в отдельном виртуальном окружении)
3. Добавьте unit-файлы в `/etc/systemd/system/` (пример ниже)
4. Настройте nginx, если нужно использовать общий домен

### Пример `systemd` для портала

`/etc/systemd/system/portal.service`
```ini
[Unit]
Description=Streamlit Portal
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/srv/portal
Environment="APPS_ROOT=/srv/portal/apps"
Environment="PORTAL_CANONICAL_ORIGIN=https://portal.internal"
ExecStart=/usr/bin/streamlit run portal_streamlit/main.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

### Пример для одной аппки

`/etc/systemd/system/promiser.service`
```ini
[Unit]
Description=pROMIser Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/srv/portal/apps/promiser
ExecStart=/usr/bin/streamlit run src/streamlit_pROMIser.py --server.port=8511
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🔁 Обновление

```bash
cd /srv/portal
git pull
sudo systemctl restart portal
sudo systemctl restart promiser
```

---

## 🧰 Логирование

```bash
journalctl -u portal -f
journalctl -u promiser -f
```

---

## ⚙️ Советы

- Убедитесь, что все порты (`8501`, `8511`, `8512`, …) открыты в `ufw` или `nginx`
- Можно добавить новые приложения просто создав новую папку с `app.yaml` и `src/*.py`
- Портал автоматически их подхватит
