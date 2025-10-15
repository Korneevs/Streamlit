#!/usr/bin/env bash
set -euo pipefail
APPS_ROOT=${APPS_ROOT:-/srv/portal/apps}

for d in "$APPS_ROOT"/*; do
  [ -d "$d" ] || continue
  slug=$(basename "$d")
  systemctl is-enabled "streamlit@${slug}" >/dev/null 2>&1 || systemctl enable --now "streamlit@${slug}"
  systemctl is-active "streamlit@${slug}" >/dev/null 2>&1 || systemctl restart "streamlit@${slug}"
done
