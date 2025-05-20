# How to run the experiments?
1. pip install -r requirements.txt
3. python olpc_lightfm.py

# Preprocessing steps
## 1. Load NAPLAN Data
1. **Source File:** `naplan_scores.csv`
2. **Action:** 
   - Read in the CSV.
   - **Average** scores per school. For each `school_id`, calculate an average NAPLAN score across subjects and year_tested.
3. **Binning:**
   - Convert the numeric `avg_score` into discrete bins (e.g. `bin1`, `bin2`, etc.) to reduce feature explosion.

## 2. Load App Usage Data
1. **Source File:** `aggregated-school-device-app-category-duration.csv`
2. **Action:** 
   - Read in the aggregated usage data.
   - Rename columns for consistency:
     - `school id` → `school_id`
     - `app id` → `app_id`
     - `total duration` → `duration`
3. **Device Count Normalization:**  
   - For each school, count the unique devices (`device_count`).
   - Merge the device count back to the usage DataFrame.
   - Aggregate by `(school_id, app_id)` and **divide total usage by `device_count`**.
   
## 3. Handle Zero Usage
1. Filter out rows where `duration` (or `duration_device_ratio`) is `0`.
2. Rationale: LightFM assumes non-existing (zero) interactions implicitly, so explicitly storing zero rows is redundant.

## 4. Transform / Scale
1. Usage distribution is extremely skewed, apply a log transform.
2. Use quantile transformer on the aggregated `norm_device_duration_ratio` as it is too skewed to the left.


# Adding a New Experiment

**Create** `experiment3.py` under `experiments/`.  
   - Define a function named `run_experiment3` that takes data objects (`train`, `test`, and optionally `user_features` and `item_features`) and returns a dictionary of metrics.

   ```python
   from lightfm import LightFM
   from lightfm.evaluation import precision_at_k, auc_score

   def run_experiment3(train, test, dataset=None, user_features=None):
       model = LightFM(loss='warp', learning_rate=0.005, random_state=42)
       if user_features is not None:
           model.fit(train, user_features=user_features, epochs=20, num_threads=4)
       else:
           model.fit(train, epochs=20, num_threads=4)

       # Evaluate
       if user_features is not None:
           prec = precision_at_k(model, test, user_features=user_features, k=5).mean()
           au = auc_score(model, test, user_features=user_features).mean()
       else:
           prec = precision_at_k(model, test, k=5).mean()
           au = auc_score(model, test).mean()

       return {'precision': prec, 'auc': au}
