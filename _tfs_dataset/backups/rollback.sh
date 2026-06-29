#!/usr/bin/env bash
# One-button rollback for the live model.
#
# Usage on the droplet:
#   /root/rollback.sh prod   # restore production /predict (Dense MLP)
#   /root/rollback.sh exp    # restore experimental /exp/predict (GRU)
#   /root/rollback.sh prod model-2026-06-04-prod-dense-8class-valacc9959
#                             # restore a specific archive instead of "the latest"
#   /root/rollback.sh list   # show what's in archive
#
# The script:
#   1) Saves whatever is currently live to .pre-rollback.<timestamp>
#   2) Copies the archive folder over the live model
#   3) Restarts the backend container so the new model is served
# It never deletes the previous live model, just renames it, so a botched
# rollback can be undone by mv-ing the .pre-rollback folder back.

set -euo pipefail

SCOPE="${1:-}"
ARCHIVE_NAME="${2:-}"

if [ -z "$SCOPE" ] || [ "$SCOPE" = "-h" ] || [ "$SCOPE" = "--help" ]; then
    cat <<EOF
Usage:
  $0 prod                          # restore latest prod archive (Dense MLP, /predict)
  $0 exp                           # restore latest exp archive (GRU, /exp/predict)
  $0 prod <archive-folder-name>    # restore a specific archive
  $0 exp  <archive-folder-name>
  $0 list                          # show available archives

Default action restores the alphabetically last archive folder, which is
the most recently dated snapshot (e.g. model-2026-06-04-...).
EOF
    exit 0
fi

if [ "$SCOPE" = "list" ]; then
    echo "--- prod archives ---"
    ls -1 /root/SIGNLANG-APP/models/archive/ 2>/dev/null || echo "  (none)"
    echo "--- exp archives ---"
    ls -1 /root/SIGNLANG-APP/models/exp/archive/ 2>/dev/null || echo "  (none)"
    exit 0
fi

case "$SCOPE" in
    prod)
        ARCHIVE_DIR=/root/SIGNLANG-APP/models/archive
        LIVE_DIR=/root/SIGNLANG-APP/models/prod/model-latest
        ;;
    exp)
        ARCHIVE_DIR=/root/SIGNLANG-APP/models/exp/archive
        LIVE_DIR=/root/SIGNLANG-APP/models/exp/prod/model-latest
        ;;
    *)
        echo "ERROR: scope must be 'prod' or 'exp' (got '$SCOPE')" >&2
        exit 1
        ;;
esac

if [ -z "$ARCHIVE_NAME" ]; then
    # pick the alphabetically last archive = most recent dated snapshot
    ARCHIVE_NAME=$(ls -1 "$ARCHIVE_DIR" 2>/dev/null | tail -n 1 || true)
    if [ -z "$ARCHIVE_NAME" ]; then
        echo "ERROR: no archives found in $ARCHIVE_DIR" >&2
        exit 1
    fi
    echo "No archive specified -> using latest: $ARCHIVE_NAME"
fi

SOURCE="$ARCHIVE_DIR/$ARCHIVE_NAME"

if [ ! -d "$SOURCE" ]; then
    echo "ERROR: archive not found: $SOURCE" >&2
    echo "Available:" >&2
    ls -1 "$ARCHIVE_DIR" >&2
    exit 1
fi

if [ ! -f "$SOURCE/model.json" ] || [ ! -f "$SOURCE/labels.json" ]; then
    echo "ERROR: archive looks incomplete (missing model.json or labels.json)" >&2
    exit 1
fi

TS=$(date +%Y-%m-%dT%H-%M-%S)
PRE_ROLLBACK="${LIVE_DIR}.pre-rollback.${TS}"

echo "  scope         = $SCOPE"
echo "  archive       = $SOURCE"
echo "  live target   = $LIVE_DIR"
echo "  backup of live -> $PRE_ROLLBACK"

if [ -d "$LIVE_DIR" ]; then
    mv "$LIVE_DIR" "$PRE_ROLLBACK"
fi
cp -r "$SOURCE" "$LIVE_DIR"

echo "Restarting backend container..."
docker restart signlang-app_backend_1 >/dev/null
sleep 3

# Health check
HTTP_CODE=$(curl -sS -k -o /dev/null -w '%{http_code}' "https://localhost/api/${SCOPE}/models" || echo "000")
if [ "$SCOPE" = "prod" ]; then
    HTTP_CODE=$(curl -sS -k -o /dev/null -w '%{http_code}' https://localhost/api/models || echo "000")
fi

echo "Health check: /api/${SCOPE}/models -> HTTP $HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "WARN: health check failed. The .pre-rollback backup is still at $PRE_ROLLBACK"
    echo "To undo this rollback:"
    echo "  rm -rf $LIVE_DIR && mv $PRE_ROLLBACK $LIVE_DIR && docker restart signlang-app_backend_1"
    exit 2
fi

echo "✓ Rollback complete. Old live model preserved at $PRE_ROLLBACK"
echo "  (Safe to delete once you have confirmed the rollback works for real users.)"
