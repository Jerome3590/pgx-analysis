# DTW Feature Analysis: Leakage Investigation

## Background: Relationship to DTW Filter Step

### Research Happens in DTW Filter (Step 4b)

**Important**: The `dtw_filter` step (Step 4b) runs **before** this `dtw_analysis` step (Step 5d). This means:

- **All research on trajectories and time windows happens in `dtw_filter`**: The filtering step analyzes all patient trajectories, time intervals, and common sequence patterns to identify what is clinically useful vs. what is noise
- **Filtering decisions are research-driven**: Based on analysis of trajectory patterns, the filter step removes protocol-like events (routine care) while preserving predictive patterns (deviations from standard care)
- **This step works with filtered data**: The `dtw_analysis` step receives already-filtered data (`model_events_no_protocols.parquet`) and extracts trajectory features from the cleaned sequences

### The Two-Step Process

1. **Step 4b (`dtw_filter`)**: 
   - Captures all trajectories with full time window information
   - Researches common sequence patterns and time intervals
   - Identifies what is protocol-like vs. clinically meaningful
   - Filters out protocol events based on research findings
   - Outputs: `model_events_no_protocols.parquet`

2. **Step 5d (`dtw_analysis`)**: 
   - Works with the filtered, cleaned data
   - Extracts trajectory features (DTW distances, trajectory characteristics)
   - Creates features for downstream modeling

**See**: `4b_dtw_filter/DTW_ROLE.md` and `4b_dtw_filter/PROTOCOL_FILTERING.md` for details on the research and filtering process.

## DTW Features Created

Based on `create_dtw_features.py`, the following features are generated:

### 1. DTW Distance Features (5-9 features)
- `dtw_distance_to_prototype_0` through `dtw_distance_to_prototype_4` (5 features)
  - Distance from patient trajectory to each of 5 prototype trajectories
- `dtw_min_distance` - Minimum distance to any prototype
- `dtw_max_distance` - Maximum distance to any prototype  
- `dtw_mean_distance` - Mean distance across all prototypes
- `dtw_std_distance` - Standard deviation of distances

### 2. Trajectory Characteristics (2 features)
- `trajectory_length` - Number of events in patient's trajectory
- `trajectory_diversity` - Number of unique activities in trajectory

**Total: ~11 features per patient**

---

## How Features Are Constructed

### Step 1: Trajectory Extraction

**Input:** `model_events.parquet` or `model_events_no_protocols.parquet` (already filtered by feature importances)

**Process:**
1. Extract events for each patient from model_data
2. **Apply cutoff dates** to exclude events after a reference point
3. Filter out F1120 codes (target event) from trajectories
4. Create sequence of activities (DRUG:*, ICD:*, CPT:*) ordered by event_date

**Cutoff Date Logic (Lines 465-489):**
```python
# For target patients: use F1120 date as cutoff (events before F1120)
# For control patients: use reference date (first event date)
```

**SQL Query:**
```sql
WITH target_cutoffs AS (
    SELECT DISTINCT
        mi_person_key,
        MIN(CASE 
            WHEN primary_icd_diagnosis_code LIKE '%F1120%' 
            THEN event_date 
            END) as cutoff_date
    FROM read_parquet('{model_data_path}')
    WHERE target = 1
    GROUP BY mi_person_key
    HAVING cutoff_date IS NOT NULL
),
control_cutoffs AS (
    SELECT 
        mi_person_key,
        MIN(event_date) as cutoff_date
    FROM read_parquet('{model_data_path}')
    WHERE target = 0
    GROUP BY mi_person_key
)
```

**Key Points:**
- ✅ Target patients: Cutoff = F1120 date (events BEFORE target event)
- ✅ Control patients: Cutoff = first event date (all events included, but consistent reference point)
- ✅ F1120 is excluded from trajectories (line 230)

### Step 2: Prototype Selection

**Process (Lines 277-296):**
1. Collect trajectories from **both target and control patients** (line 497: "combined trajectories (target + control)")
2. Sort trajectories by length
3. Select 5 prototypes evenly spaced by trajectory length (median-length approach)
4. Prototypes are selected from the **entire population**, not just targets

**Key Points:**
- ✅ Prototypes are from combined target + control trajectories
- ✅ Prototypes selected by length, not by target status
- ⚠️ **Potential Issue**: If target patients have systematically different trajectory lengths, prototypes might be biased

### Step 3: Distance Calculation

**Process (Lines 298-339):**
1. Encode all trajectories to numeric sequences
2. Compute DTW distance from each patient to each prototype
3. Calculate statistics (min, max, mean, std) across prototype distances
4. Calculate trajectory characteristics (length, diversity)

---

## Leakage Analysis

### ✅ **LEGITIMATE Features (No Leakage)**

#### 1. Trajectory Length & Diversity
- **Feature:** `trajectory_length`, `trajectory_diversity`
- **Calculation:** Based on events BEFORE cutoff date
- **Leakage Risk:** ✅ **NONE** - These are pre-target event characteristics
- **Rationale:** Count of events and unique activities before target event (targets) or reference date (controls)

#### 2. DTW Distances to Prototypes
- **Feature:** `dtw_distance_to_prototype_*`, `dtw_min/max/mean/std_distance`
- **Calculation:** Distance from patient's pre-cutoff trajectory to prototype trajectories
- **Leakage Risk:** ⚠️ **POTENTIAL** - Depends on prototype selection

**Prototype Selection Analysis:**
- Prototypes are selected from **combined target + control** trajectories (line 497)
- Prototypes selected by trajectory length, not target status
- **If prototypes are unbiased:** ✅ No leakage
- **If target patients have systematically longer/shorter trajectories:** ⚠️ Prototypes might be biased toward target patterns

---

## Potential Leakage Scenarios

### Scenario 1: Prototype Bias (MODERATE RISK)

**Issue:** If target patients have systematically different trajectory characteristics than controls, prototypes selected by length might be biased.

**Example:**
- Target patients: Average trajectory length = 50 events
- Control patients: Average trajectory length = 20 events
- Prototypes selected at length percentiles: [10, 25, 50, 75, 90]
- Result: Prototypes might be mostly from target patients

**Detection:**
```python
# Check prototype composition
prototype_patients = [trajectory_lengths[int(i * (n_patients - 1) / (n_prototypes - 1))][0] 
                      for i in range(n_prototypes)]
# Check if prototypes are mostly target patients
```

**Mitigation:**
- Select prototypes separately from target and control populations
- Use stratified prototype selection
- Verify prototype composition is balanced

### Scenario 2: Cutoff Date Inconsistency (LOW RISK)

**Issue:** Target patients use F1120 date as cutoff, controls use first event date. This creates different reference points.

**Analysis:**
- ✅ **Target patients:** Events before F1120 (predictive)
- ✅ **Control patients:** All events (but no target event, so consistent)
- ⚠️ **Potential issue:** If controls have longer histories, their trajectories might be systematically longer

**Mitigation:**
- Use fixed lookback window for both (e.g., 2 years before first event)
- Or use first event date for both (but then targets lose pre-F1120 events)

### Scenario 3: F1120 Exclusion (LEGITIMATE)

**Issue:** F1120 is excluded from trajectories (line 230).

**Analysis:**
- ✅ **Correct:** F1120 is the target event, should not be in trajectory
- ✅ **No leakage:** Trajectories end before target event

---

## Recommended Investigation Steps

### 1. Analyze Prototype Composition

**Check if prototypes are biased toward target patients:**

```python
# After prototype selection, check:
prototype_target_status = []
for proto_pid in prototype_indices:
    is_target = base_df[base_df['mi_person_key'] == proto_pid]['target'].iloc[0]
    prototype_target_status.append(is_target)

print(f"Prototype composition: {sum(prototype_target_status)}/{len(prototype_target_status)} are target patients")
```

**Expected:** If unbiased, prototypes should be roughly proportional to target/control ratio in population.

### 2. Compare Trajectory Lengths

**Check if target and control patients have different trajectory lengths:**

```python
target_lengths = [len(traj) for pid, traj in trajectories.items() 
                  if base_df[base_df['mi_person_key'] == pid]['target'].iloc[0] == 1]
control_lengths = [len(traj) for pid, traj in trajectories.items() 
                   if base_df[base_df['mi_person_key'] == pid]['target'].iloc[0] == 0]

print(f"Target avg length: {np.mean(target_lengths)}")
print(f"Control avg length: {np.mean(control_lengths)}")
```

**If significantly different:** Prototypes might be biased.

### 3. Test Feature Predictive Power

**Check if DTW features are predictive in a way that suggests leakage:**

```python
# Train model with DTW features only
# If performance is suspiciously high (>0.95 AUC), investigate leakage
# Compare performance on train vs test set
```

### 4. Verify Cutoff Dates

**Check that cutoff dates are correctly applied:**

```python
# For target patients: verify no events after F1120 date
# For control patients: verify cutoff is first event date
```

---

## Current Implementation Assessment

### ✅ **Strengths:**
1. Cutoff dates prevent using post-target events
2. F1120 is excluded from trajectories
3. Prototypes are selected from combined population
4. Trajectories are based on pre-cutoff events only

### ⚠️ **Potential Issues:**
1. **Prototype selection by length** might bias toward target patients if they have systematically different trajectory lengths
2. **Different cutoff logic** for targets (F1120 date) vs controls (first event date) might create inconsistencies
3. **No explicit balance check** on prototype composition

---

## Recommendations

### 1. Add Prototype Balance Check
- Verify prototypes are balanced between target and control
- If biased, use stratified selection

### 2. Consider Fixed Lookback Window
- Use same reference point for both targets and controls
- E.g., 2 years before first event or F1120 date (whichever is earlier)

### 3. Test Feature Performance
- Run model with DTW features only
- Check if performance suggests leakage (suspiciously high AUC)
- Compare train vs test performance

### 4. Add Logging
- Log prototype composition (target vs control)
- Log average trajectory lengths by target status
- Log cutoff date statistics

---

## Next Steps

1. **Run DTW analysis** on a test cohort to generate actual features
2. **Analyze prototype composition** to check for bias
3. **Compare trajectory characteristics** between targets and controls
4. **Test feature predictive power** to detect potential leakage
5. **Implement fixes** if leakage is detected
