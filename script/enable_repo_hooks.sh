#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$ARENA_ROOT/.githooks"

print_info() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_warning() {
    echo "[WARNING] $1"
}

run_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
    elif command -v python >/dev/null 2>&1; then
        python "$@"
    elif command -v uv >/dev/null 2>&1; then
        uv run --no-project --python 3.12 python "$@"
    else
        echo "[ERROR] Python 3 or uv is required to configure repository hooks" >&2
        return 1
    fi
}

relative_hooks_path() {
    run_python - "$1" "$2" <<'PY'
import os
import sys

hooks_dir = sys.argv[1]
repo_root = sys.argv[2]
print(os.path.relpath(hooks_dir, repo_root))
PY
}

enable_hooks_for_repo() {
    local repo_root="$1"
    local hooks_path="$2"

    if ! git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        print_warning "Skip non-git repo: $repo_root"
        return 0
    fi

    git -C "$repo_root" config --local core.hooksPath "$hooks_path"
    print_success "Enabled hooks for $repo_root -> $hooks_path"
}

main() {
    if [[ ! -f "$HOOKS_DIR/pre-commit" || ! -f "$HOOKS_DIR/commit-msg" ]]; then
        echo "[ERROR] Missing arena hook files under $HOOKS_DIR" >&2
        exit 1
    fi

    chmod +x "$HOOKS_DIR/pre-commit" "$HOOKS_DIR/commit-msg"

    enable_hooks_for_repo "$ARENA_ROOT" ".githooks"

    local child_repo
    local hooks_path
    for child_repo in "$ARENA_ROOT/submodules/IsaacLab" "$ARENA_ROOT/submodules/Isaac-GR00T"; do
        if [[ ! -d "$child_repo" ]]; then
            print_warning "Skip missing child repo: $child_repo"
            continue
        fi

        hooks_path="$(relative_hooks_path "$HOOKS_DIR" "$child_repo")"
        enable_hooks_for_repo "$child_repo" "$hooks_path"
    done
}

main "$@"
