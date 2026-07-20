# ── Stage: OpenAI Codex CLI (optional; llm.provider: codex) ───────────────────
# The CLI is a native (Rust) binary vendored inside the npm package — install
# it in a node stage and copy ONLY the binary out, so the runtime image stays
# node-free. Build with INSTALL_CODEX=false to skip entirely.
FROM node:20-slim AS codex-cli
ARG INSTALL_CODEX=true
ARG CODEX_NPM_VERSION=latest
RUN mkdir -p /out && if [ "$INSTALL_CODEX" = "true" ]; then \
        npm install -g --no-fund --no-audit "@openai/codex@${CODEX_NPM_VERSION}" \
        && ARCH="$(uname -m)" \
        && BIN="$(find /usr/local/lib/node_modules/@openai/codex \
                  -type f -name "codex-${ARCH}-*linux*" | head -n1)" \
        && BIN="${BIN:-$(find /usr/local/lib/node_modules/@openai/codex \
                  -type f -name 'codex-*linux*' | head -n1)}" \
        && test -n "$BIN" \
        && cp "$BIN" /out/codex-real && chmod +x /out/codex-real \
        && /out/codex-real --version; \
    fi

FROM python:3.11-slim

# System deps: build toolchain for numpy/pandas/Twisted wheels, tzdata for TZ,
# gosu to drop privileges in the entrypoint after fixing volume ownership.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential tzdata gosu \
    && rm -rf /var/lib/apt/lists/*

# Codex CLI (empty copy when INSTALL_CODEX=false). /usr/local/bin/codex is a
# wrapper: `docker compose exec` shells run as root, but the agent runs as
# PUID:PGID — after any root-run `codex login`, the wrapper chowns CODEX_HOME
# so the agent can read AND refresh the token without a container restart.
COPY --from=codex-cli /out/ /usr/local/bin/
RUN if [ -f /usr/local/bin/codex-real ]; then \
        printf '%s\n' \
          '#!/bin/sh' \
          '/usr/local/bin/codex-real "$@"' \
          'rc=$?' \
          'if [ "$(id -u)" = "0" ] && [ -n "$CODEX_HOME" ] && [ -d "$CODEX_HOME" ]; then' \
          '  chown -R "${PUID:-568}:${PGID:-568}" "$CODEX_HOME" 2>/dev/null || true' \
          'fi' \
          'exit $rc' \
        > /usr/local/bin/codex && chmod +x /usr/local/bin/codex; \
    fi

# PIP_DEFAULT_TIMEOUT / PIP_RETRIES: homelab networks are often slow or flaky —
# without these, one 15s index timeout aborts the whole build with a misleading
# "Could not find a version ... (from versions: none)" error.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# Install deps first for layer caching — read straight from pyproject so the
# list has a single source of truth, and copy src only AFTER installing so a
# code edit never invalidates this (download-heavy) layer. The dashboard extra
# is included so the same image runs either the agent or the dashboard.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && python -c "import tomllib; p = tomllib.load(open('pyproject.toml','rb'))['project']; print(chr(10).join(p['dependencies'] + p['optional-dependencies']['dashboard']))" > /tmp/deps.txt \
 && pip install -r /tmp/deps.txt \
 && rm /tmp/deps.txt

# Source last; deps already present, so this layer is quick and cache-friendly.
COPY src ./src
RUN pip install --no-deps -e ".[dashboard]"

COPY config ./config
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Normalize COPYed app code/config perms so any runtime UID (we gosu-drop to
# PUID:PGID, which owns neither the files nor their group) can read and import
# them. The image otherwise inherits whatever modes the build host had — and a
# restrictive dataset umask/ACL (common on TrueNAS) can strip the world-read
# bit, causing `PermissionError` on import. `a+rX` adds read everywhere +
# traverse on dirs. (site-packages is installed by pip as root with normal
# world-readable modes, so it needs no fixup — and skipping it keeps the build
# from recursing over the whole numpy/pandas tree.)
RUN chmod -R a+rX /app/src /app/config /app/pyproject.toml

# Mutable runtime state (journal DB, status/control, token, learned params).
# Persisted via the data volume; made group-writable so the container can run as
# an arbitrary non-root UID/GID (e.g. TrueNAS "apps" 568) against a host dataset.
RUN mkdir -p /app/data && chmod -R 0775 /app/data

# NOTE: no `VOLUME ["/app/data"]` here on purpose. A Dockerfile VOLUME creates a
# fresh *anonymous* volume whenever the container is recreated (e.g. on
# `docker compose up -d --build`) unless an explicit mount shadows it — a common
# cause of "my data resets on rebuild". Persistence is defined explicitly in
# docker-compose.yml (named volume `gungnir-data`, or a bind mount), which the
# entrypoint chowns to PUID:PGID before `gosu` drops privileges.

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "gungnir.main", "--config", "config/config.yaml"]
