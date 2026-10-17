#!/usr/bin/env bash
# Выкладка сводки на VPS с Ubuntu 24.04: bash news/deploy.sh <IP>. Повторный запуск обновляет код.
# Ставит Python-окружение, службу systemd и Caddy — HTTPS на адресе <IP через дефисы>.sslip.io.
set -euo pipefail
IP=${1:?укажите IP сервера}
HOST="${IP//./-}.sslip.io"
SSH=(ssh -i "$HOME/.ssh/id_ed25519_h3llo" -o StrictHostKeyChecking=accept-new "user@$IP")
cd "$(dirname "$0")/.."

tar -czf - --exclude=__pycache__ news | "${SSH[@]}" "sudo mkdir -p /opt/news && sudo tar -xzf - --no-same-owner -C /opt/news"
# на сервер — только ключ модели, в файл, который читает один root; служба получает его через EnvironmentFile
if ! "${SSH[@]}" "sudo test -s /etc/news.env"; then
  grep '^DASHSCOPE_API_KEY=' .env | tr -d '\r' | "${SSH[@]}" "sudo sh -c 'umask 077; cat > /etc/news.env'"
fi

"${SSH[@]}" "sudo bash -s" <<EOF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v caddy >/dev/null || { apt-get update -q; apt-get install -yq python3-venv caddy; }
cd /opt/news
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r news/requirements.txt
cat > /etc/systemd/system/news.service <<'UNIT'
[Unit]
Description=Svodka - daily news digest
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/news
EnvironmentFile=/etc/news.env
Environment=HOST=127.0.0.1 PORT=8000 DATA_DIR=/var/lib/news COOKIE_SECURE=1 PYTHONUNBUFFERED=1
ExecStart=/opt/news/.venv/bin/python -m news.main
DynamicUser=yes
StateDirectory=news
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/caddy/Caddyfile <<'CADDY'
$HOST {
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1
    }
}
CADDY
systemctl daemon-reload
systemctl enable --quiet news
systemctl restart news
systemctl reload caddy || systemctl restart caddy
EOF
echo "Готово: https://$HOST"
