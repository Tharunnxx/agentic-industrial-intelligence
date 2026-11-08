# GitHub setup for this project

## Important

The application source code and runtime paths are unchanged. The repository keeps the live
`models/cost_sensitive_model.joblib` artifact because the backend loads it from that path by default.

The model is large, so use Git LFS before the first commit:

```powershell
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes
```

The original SCANIA raw dataset is intentionally not included. Keep the dataset in its existing
external location and set `SCANIA_DATA_DIR` when running the application.

## First push

From the `agentic_platform` directory:

```powershell
git init
git lfs install
git add .
git status
git commit -m "Initial capstone project"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

Before committing, check `git status` and make sure no `.env`, raw dataset, or uploaded dataset is staged.

## Local application

The existing `run_demo.ps1`, backend, frontend, model path, configuration, samples, tests and
research files are preserved. No application code was changed as part of the GitHub cleanup.
