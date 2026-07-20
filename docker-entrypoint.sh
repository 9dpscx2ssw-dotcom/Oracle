#!/usr/bin/env bash
# Entrypoint that makes the mounted /app/data volume writable, then drops to an
# unprivileged user before running the app.
#
# Why this exists: a bind-mounted host directory keeps the *host's* ownership,
# which shadows any `chown`/`chmod` baked into the image. So when the container
# runs as UID 568 against a dataset owned by root (or another UID), writes to
# /app/data fail with EACCES. Running the entrypoint as root lets us chown the
# volume to PUID:PGID at startup, then `gosu` steps down to that user so the app
# itself never runs as root.
#
# PUID/PGID default to 568 (the TrueNAS "apps" user). Override in .env.
set -euo pipefail

PUID="${PUID:-568}"
PGID="${PGID:-568}"
DATA_DIR="/app/data"
# App code lives in the image; the dropped user must be able to read/import it.
CODE_PATHS="/app/src /app/pyproject.toml"

# Only root can fix ownership / step down. If compose pinned `user:` we're not
# root — skip straight to running, relying on the host dir already being correct.
if [ "$(id -u)" = "0" ]; then
    # Ensure a group + user exist for the requested IDs so gosu can resolve them.
    if ! getent group "$PGID" >/dev/null 2>&1; then
        groupadd --gid "$PGID" gungnir 2>/dev/null || true
    fi
    if ! getent passwd "$PUID" >/dev/null 2>&1; then
        useradd --uid "$PUID" --gid "$PGID" --no-create-home --shell /usr/sbin/nologin gungnir 2>/dev/null || true
    fi

    mkdir -p "$DATA_DIR"
    # Best-effort: take ownership of the volume so the app can write its journal
    # DB, status/control files, token, and learned params.
    if ! chown -R "$PUID:$PGID" "$DATA_DIR" 2>/dev/null; then
        echo "WARNING: could not chown $DATA_DIR to $PUID:$PGID — check host dataset permissions." >&2
    fi
    chmod -R u+rwX,g+rwX "$DATA_DIR" 2>/dev/null || true

    # Make the dropped user the OWNER of the app code. POSIX mode bits alone
    # aren't enough on TrueNAS: the Docker storage dataset may carry NFSv4 ACLs
    # that ignore the world-read bit, so a non-owner UID gets EACCES importing
    # /app/src even at mode 0644. owner@ access *is* honored under those ACLs,
    # and chowning as root recomputes the ACL — so own the code, then read it.
    chown "$PUID:$PGID" /app 2>/dev/null || true   # traverse into /app
    chown -Rh "$PUID:$PGID" $CODE_PATHS 2>/dev/null || true
    chmod -R u+rX $CODE_PATHS 2>/dev/null || true

    exec gosu "$PUID:$PGID" "$@"
fi

# Non-root (compose set `user:`). Warn early if the volume isn't writable.
if [ ! -w "$DATA_DIR" ]; then
    echo "WARNING: $DATA_DIR is not writable by UID $(id -u). On the host run:" >&2
    echo "         chown -R ${PUID}:${PGID} <your DATA_DIR> && chmod -R 775 <your DATA_DIR>" >&2
fi

exec "$@"
