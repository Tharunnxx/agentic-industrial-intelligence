# models/

Not used by default: the platform loads your existing `cost_sensitive_model.joblib` from the parent project folder
(read-only). Put a copy of an artifact here only if you want a self-contained folder, then set
`AGENTIC_MODEL_PATH=agentic_platform\models\<file>.joblib` in `.env`.
