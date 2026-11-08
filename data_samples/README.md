# data_samples/

Empty until you run `python scripts/extract_samples.py`. It writes `sample_<vehicle_id>_class<k>.json` files built from
**real rows** of your `validation_*.csv` files (latest readout per vehicle, selected by seed, not by prediction).
No synthetic vehicle data is ever generated or shipped. Check the dataset licence before publishing these files
(`.gitignore` excludes them by default).
