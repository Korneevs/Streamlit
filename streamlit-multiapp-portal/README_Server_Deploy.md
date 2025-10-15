# Streamlit Multi‑App Portal — Server Deploy Guide

This guide explains how to deploy the multi‑app Streamlit **portal** (dispatcher) on a Linux server,
and how to run any number of Streamlit apps that the portal auto‑discovers from `apps/*/app.yaml`.

> Result: users open **one URL** (the portal), and each app renders inside an iframe without opening new tabs.

---

## 0) Overview

- **Portal URL (example):** `https://msk.avito-streamlit.ru:8501`
- **Apps location on server:** `/srv/portal/apps/<slug>/`
- **Auto‑discovery:** each app has an `app.yaml` with at least: `slug`, `title`, `port`, `entry`
- **Isolation:** each app runs as its own Streamlit process **on its own port**
- **Embedding:** the portal embeds apps from the **same host** using the port from `app.yaml`

---

## 1) Prerequisites

- Linux host (Ubuntu/Debian/RHEL ok)
- Python **3.9+**
- `sudo` access
- DNS set to your server (e.g. `msk.avito-streamlit.ru`)
- (Optional) Corporate VPN / firewall already configured

---

## 2) Directory layout on the server

```text
/srv/portal/
  ├─ portal_streamlit/
  │   └─ main.py                 # the portal (dispatcher)
  ├─ apps/                       # your Streamlit apps here (any number)
  │   ├─ promiser/
  │   │   ├─ app.yaml
  │   │   ├─ requirements.txt
  │   │   └─ src/streamlit_app.py
  │   └─ matrexa/
  │       ├─ app.yaml
  │       ├─ requirements.txt
  │       └─ src/streamlit_app.py
  ├─ requirements.txt            # portal dependencies (streamlit + pyyaml)
  └─ scripts/
      └─ ensure_apps.sh          # optional helper to (re)start app services
```

**Example `app.yaml` (per app):**
```yaml
slug: promiser
title: "pROMIser"
description: "DTB prediction"
icon: "📈"
port: 8511
entry: "src/streamlit_app.py"
```

> Use **unique ports** per app (8511, 8512, 8513, …).

---

## 3) Install portal (one time)

```bash
sudo mkdir -p /srv/portal
sudo chown -R $USER:$USER /srv/portal

# Copy your repo files into /srv/portal (portal_streamlit/, apps/, requirements.txt, scripts/, etc.)

cd /srv/portal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt    # installs streamlit and pyyaml for the portal
```

> If your apps require heavy libs, keep those in each app's own `requirements.txt` \
> and install separately (see §5).

---

## 4) Run the portal as a systemd service

Create `/etc/systemd/system/streamlit-portal.service`:

```ini
[Unit]
Description=Streamlit Multi-App Portal
After=network.target

[Service]
WorkingDirectory=/srv/portal
Environment=PORTAL_CANONICAL_ORIGIN=https://msk.avito-streamlit.ru:8501
Environment=APPS_ROOT=/srv/portal/apps
ExecStart=/srv/portal/.venv/bin/streamlit run portal_streamlit/main.py \
  --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable & start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now streamlit-portal
sudo systemctl status streamlit-portal
```

**Logs:**
```bash
journalctl -u streamlit-portal -f
```

---

## 5) Run each app as its own service

For each app, create a unit. Example for `promiser` on port **8511**:

`/etc/systemd/system/streamlit-promiser.service`
```ini
[Unit]
Description=Streamlit app: promiser
After=network.target

[Service]
WorkingDirectory=/srv/portal/apps/promiser
# install/upgrade app deps on boot (optional but handy):
ExecStartPre=/srv/portal/.venv/bin/python -m pip install -r requirements.txt
ExecStart=/srv/portal/.venv/bin/streamlit run src/streamlit_app.py \
  --server.port=8511 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Repeat for `matrexa` (port **8512**) with its own paths and entry.

Enable & start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now streamlit-promiser
sudo systemctl enable --now streamlit-matrexa
```

**Check listening ports:**
```bash
ss -tulpn | grep 85
```

---

## 6) Optional: auto‑(re)start helpers

A simple helper can ensure units are enabled/active whenever a new app folder appears.

`/srv/portal/scripts/ensure_apps.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
APPS_ROOT=${APPS_ROOT:-/srv/portal/apps}

for d in "$APPS_ROOT"/*; do
  [ -d "$d" ] || continue
  slug=$(basename "$d")
  svc="streamlit-${slug}.service"
  if systemctl list-unit-files | grep -q "^${svc}"; then
    systemctl is-enabled "$svc" >/dev/null 2>&1 || sudo systemctl enable --now "$svc"
    systemctl is-active "$svc"  >/dev/null 2>&1 || sudo systemctl restart "$svc"
  else
    echo "Unit ${svc} not found. Create it in /etc/systemd/system/"
  fi
done
```

Make it executable and (optionally) cron it:
```bash
sudo chmod +x /srv/portal/scripts/ensure_apps.sh
# every 5 minutes
sudo crontab -e
*/5 * * * * APPS_ROOT=/srv/portal/apps /srv/portal/scripts/ensure_apps.sh
```

---

## 7) Nginx (optional)

You **do not need** Nginx if users will access host:ports via VPN.  
If you prefer a single domain without visible ports, add a path‑proxy like this:

```nginx
server {
  listen 80;
  server_name msk.avito-streamlit.ru;
  return 301 https://$host$request_uri;
}

server {
  listen 443 ssl http2;
  server_name msk.avito-streamlit.ru;
  # ssl_certificate /etc/ssl/certs/your.crt;
  # ssl_certificate_key /etc/ssl/private/your.key;

  # portal
  location / {
    proxy_pass http://127.0.0.1:8501/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  # apps (example slugs)
  location ^~ /apps/promiser/ { proxy_pass http://127.0.0.1:8511/; }
  location ^~ /apps/matrexa/  { proxy_pass http://127.0.0.1:8512/; }

  # allow embedding
  add_header X-Frame-Options SAMEORIGIN always;
  add_header Content-Security-Policy "frame-ancestors 'self'" always;

  # WebSocket
  proxy_http_version 1.1;
  map $http_upgrade $connection_upgrade { default upgrade; '' close; }
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection $connection_upgrade;
}
```

> If you switch to path‑proxy, change the portal’s iframe `src` to use `/apps/<slug>/` instead of ports.

---

## 8) Updates & zero‑downtime tips

- Pull new code and restart services:
  ```bash
  cd /srv/portal && git pull
  sudo systemctl restart streamlit-portal
  sudo systemctl restart streamlit-promiser
  # ... other app units
  ```
- Update only an app’s dependencies without dropping the portal:
  ```bash
  sudo systemctl stop streamlit-promiser
  source /srv/portal/.venv/bin/activate && pip install -r /srv/portal/apps/promiser/requirements.txt
  sudo systemctl start streamlit-promiser
  ```
- Use different ports per app to avoid collisions.

---

## 9) Troubleshooting

- **Portal shows “No apps yet”**  
  Check env of the portal unit: `APPS_ROOT=/srv/portal/apps`  
  Each app folder must contain a valid `app.yaml` with `slug`, `title`, `port`, `entry`.

- **Iframe is blank / WebSocket errors**  
  Ensure app ports are reachable from the portal host. With Nginx, keep `proxy_http_version 1.1` and upgrade headers.

- **Redirect loop to another URL**  
  Set `PORTAL_CANONICAL_ORIGIN` correctly in the portal unit
  (prod example: `https://msk.avito-streamlit.ru:8501`).

- **`unhashable type` when reading YAML**  
  Ensure the portal code uses normal `{}` (not doubled `{{}}`) and that your YAML is valid.

- **Port already in use**  
  Pick a free port in `app.yaml`, update the app unit, `daemon-reload`, restart.

---

## 10) Security notes

- Keep the portal and app ports **internal** (VPN/firewall) or put Nginx in front.
- Use **same-origin** embedding (portal and apps under the same host) to avoid cross‑origin issues.
- Limit access at your network layer (VPN, allow‑lists).

---

## 11) Quick commands recap

```bash
# Portal
sudo systemctl enable --now streamlit-portal
journalctl -u streamlit-portal -f

# Apps
sudo systemctl enable --now streamlit-promiser
sudo systemctl enable --now streamlit-matrexa
journalctl -u streamlit-promiser -f

# Ports
ss -tulpn | grep 85
```
