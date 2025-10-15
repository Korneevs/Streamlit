
import os
import glob
import yaml
import streamlit as st

# === SETTINGS ===
# Canonical URL for the portal (must include scheme + host + :port). Example: "https://msk.avito-streamlit.ru:8501"
PORTAL_CANONICAL_ORIGIN = os.environ.get("PORTAL_CANONICAL_ORIGIN", "https://msk.avito-streamlit.ru:8501")
# Folder with apps (each with app.yaml)
APPS_ROOT = os.environ.get("APPS_ROOT", "/srv/portal/apps")

st.set_page_config(page_title="🧭 Внутренние приложения", layout="wide")
st.title("🧭 Внутренние Streamlit‑приложения")

# Keep users on the canonical URL
st.components.v1.html(f"""
<script>
(function() {{
  try {{
    var canonical = "{PORTAL_CANONICAL_ORIGIN}";
    if (canonical.endsWith('/')) canonical = canonical.slice(0, -1);
    var here = window.location.origin + window.location.pathname;
    var want = canonical + "/";
    if (here !== want) {{
      var target = canonical + window.location.search + window.location.hash;
      if (window.top === window.self) window.location.replace(target);
      else window.top.location.replace(target);
    }}
  }} catch(e) {{ /* noop */ }}
}})();
</script>
""", height=0)

# Load all app.yaml files
apps = []
for yml_path in sorted(glob.glob(os.path.join(APPS_ROOT, "*", "app.yaml"))):
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}          # <= одинарные скобки
        required = {"slug", "title", "port"}        # <= множество строк
        if not required.issubset(meta.keys()):
            continue
        apps.append({                               # <= одинарные скобки
            "slug": str(meta["slug"]),
            "title": str(meta["title"]),
            "description": str(meta.get("description") or ""),
            "icon": str(meta.get("icon") or "💡"),
            "port": int(meta["port"]),
            "entry": str(meta.get("entry") or "streamlit_app.py"),
        })
    except Exception as e:
        st.warning(f"Не удалось прочитать {yml_path}: {e}")


if not apps:
    st.info("Пока нет приложений. Добавьте папки с app.yaml в /srv/portal/apps/.")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.subheader("Приложения")
    options = [a["slug"] for a in apps]
    selected_slug = st.radio(
        label="",
        options=options,
        format_func=lambda s: next((f'{a.get("icon","💡")}  {a["title"]}' for a in apps if a["slug"]==s), s),
    )
    selected = next(a for a in apps if a["slug"] == selected_slug)
    if selected.get("description"):
        st.caption(selected["description"])

# Build iframe src: same host as portal, different port from YAML
def _scheme_host(origin: str) -> str:
    # https://host:port -> https://host
    if origin.startswith("http"):
        try:
            scheme, rest = origin.split("://", 1)
            host = rest.split(":")[0]
            return f"{scheme}://{host}"
        except Exception:
            return origin
    return origin.rstrip("/")

APP_PUBLIC_ORIGIN = _scheme_host(PORTAL_CANONICAL_ORIGIN)
src = f"{APP_PUBLIC_ORIGIN}:{selected['port']}/"

# Render iframe without allow-popups so apps don't open new tabs
st.components.v1.html(
    f'''
<div style="height:900px; border:0; padding:0; margin:0;">
  <iframe
    src="{src}"
    style="width:100%; height:100%; border:0;"
    sandbox="allow-scripts allow-same-origin allow-forms"
    referrerpolicy="no-referrer"
    allow="clipboard-read; clipboard-write"
  ></iframe>
</div>
''',
    height=900,
)

st.caption(f"Портал: {PORTAL_CANONICAL_ORIGIN} • Приложение: {src}")
