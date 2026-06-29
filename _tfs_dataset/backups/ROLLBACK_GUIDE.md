# Quick Rollback Guide

What to do if real users report the app is acting up. All commands assume you
are SSH'd into the droplet:

```bash
ssh root@139.59.57.147
```

## Case 1 — Production /predict is wrong / broken

```bash
/root/rollback.sh prod          # restores the latest backup
# OR
/root/rollback.sh prod model-2026-06-04-prod-dense-8class-valacc9959
```

The script:
1. Renames the live model to `.pre-rollback.<timestamp>` (kept as safety net)
2. Copies the chosen archive over
3. Restarts the backend
4. Runs a quick HTTP health check and reports the result

Then the user refreshes https://139-59-57-147.nip.io/predict and it's restored.

## Case 2 — Experimental /exp/predict is wrong / broken

```bash
/root/rollback.sh exp
```

Affects only the GRU model. Production /predict is untouched.

## Case 3 — Both models look fine, but the website itself broke

That is a frontend or backend code issue, not a model issue. Two-step fix on
your local machine:

```bash
cd d:/FromE/Bank/Project/signlang-app/frontend
git log --oneline main -5            # find the last known-good commit
git checkout <good-commit-hash>      # detached HEAD is fine
npm run build
scp -r dist root@139.59.57.147:/root/SIGNLANG-APP/frontend/dist.new
ssh root@139.59.57.147 "cd /root/SIGNLANG-APP/frontend && rm -rf dist.bak && mv dist dist.bak && mv dist.new dist && docker restart signlang-app_frontend_nginx_1"
```

Same pattern for backend repo if it was the backend code that broke (replace
`frontend` with `backend`, `signlang-app_frontend_nginx_1` with
`signlang-app_backend_1`, and use `docker cp` instead of mounting dist).

## Case 4 — A rollback itself broke something

Every rollback keeps the previous live model at
`...model-latest.pre-rollback.<timestamp>/`. To undo:

```bash
# For prod:
rm -rf /root/SIGNLANG-APP/models/prod/model-latest
ls -d /root/SIGNLANG-APP/models/prod/model-latest.pre-rollback.* | tail -1 | \
   xargs -I{} mv {} /root/SIGNLANG-APP/models/prod/model-latest
docker restart signlang-app_backend_1

# For exp: swap "models/prod" with "models/exp/prod"
```

## What you cannot accidentally break

- End users testing the app cannot promote a model to prod. The
  `/api/promote-model` endpoint exists in the backend but is not wired into
  the TrainingPage UI. Reaching it requires a manual `curl POST`.
- The TrainingPage saves models into `dev/`, never directly into `prod/`.
- The frontend dist swap is atomic (`mv` then docker restart) — the user
  sees either the old build or the new build, never a half-loaded mix.
- The rollback script never deletes anything before the new model is in
  place. The old live model survives as `.pre-rollback.<timestamp>` until
  you delete it manually.

## Where backups live

| Location | Purpose |
|---|---|
| `/root/SIGNLANG-APP/models/archive/` (droplet) | Production model snapshots |
| `/root/SIGNLANG-APP/models/exp/archive/` (droplet) | Experimental model snapshots |
| `d:\FromE\Bank\Project\signlang-app\_tfs_dataset\backups\` (your laptop) | Mirrored copy of both, for off-droplet redundancy |
| GitHub `main` branch (each repo) | Code history — pair with `git checkout` |

## Daily check (optional)

If you want to verify the current models are still serving correctly without
asking a user:

```bash
curl -sS https://139-59-57-147.nip.io/api/models | head -c 200
curl -sS https://139-59-57-147.nip.io/api/exp/models | head -c 200
```

Both should return a JSON array starting with the `prod/model-latest` and
`exp/prod/model-latest` entry respectively.
