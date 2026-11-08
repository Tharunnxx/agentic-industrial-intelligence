SCANIA Operations Intelligence — Frontend finalization update

Contents
- index.html
- app.css
- app.js

Changes in this update
1. Overview canvas charts now support pointer hover tooltips with category, count and share.
2. Vehicle Analysis restores the selected vehicle, feature payload, workflow result, active report tab, filters and pagination when navigating between hash-routed pages in the same page session. It also adds feature-name search and highlights the selected vehicle.
3. Advanced Input supports selecting CSV files, ZIP archives, and (in browsers supporting webkitdirectory) a dataset folder. Files are posted to the existing POST /uploads endpoint using its existing `files` field. The backend accepts the selected files; folder structure is not relied on.
4. Advanced Input now renders the workflow response as a readable summary with decision, priority, calibrated failure probability, expected costs, agent-report tabs and verification checks. The raw response remains available in a technical accordion.
5. Model Analytics now uses the existing GET /model/evaluate?split= endpoint to render the measured expected-cost policy confusion matrix and per-class metrics, actual and raw-prediction class distributions, plus the existing saved model-comparison rows.
6. Model Analytics explains class imbalance, zero recall, cost-vs-classification tradeoffs, and validation/test separation.
7. API requests have a 120-second timeout with a readable error message.

No backend, model, calibration, cost matrix, dataset, or ML decision logic is changed by this frontend package.

Important technical notes
- Folder upload uses the browser's webkitdirectory capability. This is supported by Chromium-based browsers. The backend receives the files as one multipart `files` collection; it does not receive a true directory object.
- The /model/evaluate endpoint reads the labelled split and evaluates the saved model. On large SCANIA CSVs this may take time. The UI allows up to 120 seconds per request and displays errors if evaluation cannot complete.
- Confusion matrix and per-class metrics shown are for the expected-cost policy returned by /model/evaluate. The API also returns raw_argmax metrics; this UI does not mislabel raw model scores as calibrated probabilities.
- Vehicle Analysis state is preserved while navigating within the same loaded browser tab. A full page reload resets in-memory state.
- No browser end-to-end run against the actual local backend was possible in this build environment. The container's Chromium launch was blocked by its administrator policy. Do not interpret static checks as proof of local API/browser acceptance.

Install
1. Stop FastAPI (Ctrl+C).
2. Back up your existing frontend folder.
3. Extract this archive into your existing project root, preserving `frontend/`.
4. Replace the existing frontend files when prompted.
5. Restart with: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
6. Open http://127.0.0.1:8000/ and press Ctrl+F5.
