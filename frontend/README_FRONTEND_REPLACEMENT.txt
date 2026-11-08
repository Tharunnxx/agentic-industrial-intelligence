SCANIA Operations Intelligence — Frontend replacement

This archive contains the redesigned frontend only:
  frontend/index.html
  frontend/app.css
  frontend/app.js

How to apply on Windows:
1. Stop the running FastAPI application (Ctrl+C in its terminal).
2. Make a backup copy of your current project\frontend folder.
3. Extract this ZIP into your existing project root, preserving the frontend\ folder structure.
4. Allow the three files to replace the existing frontend files.
5. Restart using:
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
6. Open http://127.0.0.1:8000/ and press Ctrl+F5 for a hard refresh.

The frontend uses the existing vanilla JavaScript application and existing FastAPI endpoints.
No backend, model, calibration, cost matrix, dataset, or ML logic was modified.

Important:
- The Maintenance Decisions page requires a completed workflow in the current browser session.
- Recent analysis history is browser-session-only; it is not persisted by the backend.
- Evaluation-only recorded labels are shown separately after the workflow and are never sent to agents.
- “Verified” refers to software checks only, not model correctness or physical equipment safety.
- The exact cost matrix is not duplicated in the frontend; the page displays the expected costs and cost breakdown returned by the backend.

Validation performed in the build environment:
- node --check frontend/app.js: passed
- python -m pytest tests -q: 114 passed
- HTTP smoke checks: /, /health, /index/status, /analytics/validation returned successfully

A full interactive browser acceptance test was not completed in this environment. Please check your local browser after replacing the files.
