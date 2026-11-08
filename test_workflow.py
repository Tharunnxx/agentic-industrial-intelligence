import json, urllib.request
d = json.load(open("data_samples/sample_5709_class0.json"))
payload = {k: d[k] for k in ("vehicle_id", "time_step", "features")}
req = urllib.request.Request("http://127.0.0.1:8000/workflow/run", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req))
json.dump(r, open("workflow_sample_5709.json", "w"), indent=1)
print("status:", r["status"], "| final:", r["final_status"], "| replans:", r["replan_count"])
print("probabilities:", {k: round(v, 4) for k, v in r["prediction"]["probabilities"].items()})
print("argmax class:", r["prediction"]["predicted_class_argmax"], "| cost-based class:", r["recommendation"]["recommended_class"], "| priority:", r["recommendation"]["priority"])
print("true label (evaluation only):", d["meta"]["evaluation_only_label"])
