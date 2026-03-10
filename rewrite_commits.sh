#!/usr/bin/env bash
# =============================================================================
# rewrite_commits.sh
# Automatically rewrites 7 commit messages via interactive rebase (no editor).
# Compatible with bash 3.2 (macOS default).
# Uses git stash --all to handle tracked, untracked, AND gitignored files.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

if ! git rev-parse --git-dir &>/dev/null; then
  error "Not inside a git repository."; exit 1
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "HEAD")
info "Working on branch: $BRANCH"

# Save rollback point
ROLLBACK_REF=$(git rev-parse HEAD)
echo "$ROLLBACK_REF" > /tmp/rewrite_commits_rollback.txt
info "Rollback ref saved: $ROLLBACK_REF"

# --------------- Stash EVERYTHING (tracked + untracked + gitignored) --------
# --all ensures git rebase --root sees a completely clean working tree,
# avoiding conflicts with untracked/ignored files in data/ and models/.
STASH_LABEL="pre-rebase-$(date +%Y%m%d-%H%M%S)"
info "Stashing all files (including gitignored) with label: $STASH_LABEL"
git stash push --all -m "$STASH_LABEL"
STASH_SHA=$(git stash list | head -1 | awk '{print $1}' | tr -d ':')
info "Stash created: $STASH_SHA"

# --------------- Rollback function ------------------------------------------
rollback() {
  error "Something went wrong. Rolling back..."
  git rebase --abort 2>/dev/null || true
  git reset --hard "$ROLLBACK_REF"
  info "Restoring stash..."
  git stash pop || warn "Could not auto-pop stash. Run:  git stash pop"
  error "Rolled back to $ROLLBACK_REF"
  exit 1
}
trap rollback ERR

# --------------- Build the GIT_SEQUENCE_EDITOR script ----------------------
# Plain case/esac — no declare -A — works on bash 3.2 (macOS default).
SEQ_EDITOR=$(mktemp /tmp/seq_editor.XXXXXX)
chmod +x "$SEQ_EDITOR"

cat > "$SEQ_EDITOR" << 'SEEOF'
#!/usr/bin/env bash
# Receives the rebase todo file as $1.
# Inserts "exec git commit --amend -m" after each target pick line.

TODO="$1"
TMP=$(mktemp)

get_new_message() {
  case "$1" in
    9c20ccb) echo "Add configuration management with YAML and environment variables" ;;
    7065094) echo "Implement GitHub API integration with PyGithub authentication" ;;
    29a2122) echo "Build data ingestion pipeline: extract 200 commits from pandas repository" ;;
    c4c5df4) echo "Implement keyword-based label generation for bug detection" ;;
    12ea7ad) echo "Engineer 22 features: commit metrics, developer history, temporal patterns" ;;
    bc22787) echo "Train baseline Logistic Regression model (75% accuracy, 68% recall)" ;;
    741e508) echo "Train XGBoost model: 80% accuracy, 74% recall (production model)" ;;
    *)       echo "" ;;
  esac
}

while IFS= read -r line; do
  printf '%s\n' "$line" >> "$TMP"
  SHORT_HASH=$(printf '%s' "$line" | awk '/^pick / {print $2}')
  if [ -n "$SHORT_HASH" ]; then
    NEW_MSG=$(get_new_message "$SHORT_HASH")
    if [ -n "$NEW_MSG" ]; then
      printf 'exec git commit --amend -m "%s"\n' "$NEW_MSG" >> "$TMP"
    fi
  fi
done < "$TODO"

cp "$TMP" "$TODO"
rm -f "$TMP"
SEEOF

# --------------- Run the rebase ---------------------------------------------
info "Starting automated interactive rebase --root ..."
GIT_SEQUENCE_EDITOR="$SEQ_EDITOR" git rebase -i --root

rm -f "$SEQ_EDITOR"
info "Rebase complete."

# --------------- Verify -----------------------------------------------------
echo ""
info "New commit log:"
git log --oneline
echo ""

# --------------- Force push -------------------------------------------------
info "Force-pushing to origin/$BRANCH ..."
if git push --force-with-lease origin "$BRANCH"; then
  info "Force push successful."
else
  warn "--force-with-lease rejected. Run manually if certain:"
  warn "  git push --force origin $BRANCH"
fi

# --------------- Restore everything from stash ------------------------------
info "Restoring all stashed files..."
git stash pop
info "Stash restored."

trap - ERR
echo ""
info "Done! 7 commit messages rewritten and pushed."
info "Original HEAD: $ROLLBACK_REF"
