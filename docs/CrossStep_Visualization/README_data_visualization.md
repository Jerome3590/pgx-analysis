## Data Visualization README

This guide shows how to build frequency datasets for `drug_name` and `target_code`, generate static charts, and publish interactive Plotly dashboards to S3. All outputs are written to S3 as Parquet/CSV with a stable "latest" key, so you can always fetch the most recent data based on S3 object metadata.

### Prerequisites
- S3 credentials available to DuckDB (`CALL load_aws_credentials()`)
- Python environment with `duckdb`, `pandas`, `matplotlib`, `seaborn`, and Plotly (Plotly is loaded from CDN in the dashboard HTML)
- Repository path added to `sys.path` in notebooks when importing helpers

Environment note: the canonical local outputs directory used by helpers is
`1_apcd_input_data/outputs` by default. You can override this per-environment
by setting the `PGX_TARGET_OUTPUTS_DIR` environment variable to an absolute
or relative path (useful on EC2 or CI):

```bash
export PGX_TARGET_OUTPUTS_DIR=/home/pgx3874/pgx-analysis/1_apcd_input_data/outputs
```

---

## 1) Build Frequency Datasets (latest Parquet/CSV in S3)

### A. drug_name frequency by year (Pharmacy)
```python
import duckdb, pandas as pd

con = duckdb.connect(database=':memory:')
con.sql("LOAD httpfs;")
con.sql("LOAD aws;")
con.sql("CALL load_aws_credentials();")
con.sql("SET s3_region='us-east-1'")
con.sql("SET s3_url_style='path'")

drug_freq = con.sql("""
SELECT
  event_year::INT AS event_year,
  drug_name,
  COUNT(*)::BIGINT AS frequency
FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band=*/event_year=*/pharmacy_data.parquet')
WHERE drug_name IS NOT NULL AND drug_name <> ''
  AND event_year BETWEEN 2016 AND 2020
GROUP BY event_year, drug_name
""").df()

from helpers_1997_13.visualization_utils import write_drug_frequency_latest
write_drug_frequency_latest(drug_freq)
```

### B. target_code frequency by year (ICD + CPT from Medical)
Use `1_apcd_input_data/6_target_frequency_analysis.py` to generate unified ICD/CPT counts and write to `s3://pgxdatalake/gold/target_code/target_code_latest.(parquet|csv)`.

Minimal notebook cell:
```python
import duckdb, pandas as pd

con = duckdb.connect(database=':memory:')
con.sql("LOAD httpfs;")
con.sql("LOAD aws;")
con.sql("CALL load_aws_credentials();")
con.sql("SET s3_region='us-east-1'")
con.sql("SET s3_url_style='path'")

# Build ICD aggregated
icd = con.sql("""
WITH icd_raw AS (
  SELECT event_year, primary_icd_diagnosis_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet')
  WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, two_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE two_icd_diagnosis_code   IS NOT NULL AND two_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, three_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, four_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE four_icd_diagnosis_code  IS NOT NULL AND four_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, five_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE five_icd_diagnosis_code  IS NOT NULL AND five_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, six_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE six_icd_diagnosis_code   IS NOT NULL AND six_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, seven_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE seven_icd_diagnosis_code IS NOT NULL AND seven_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, eight_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE eight_icd_diagnosis_code IS NOT NULL AND eight_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, nine_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE nine_icd_diagnosis_code  IS NOT NULL AND nine_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, ten_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE ten_icd_diagnosis_code   IS NOT NULL AND ten_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
)
SELECT event_year::INT AS event_year, target_code, COUNT(*)::BIGINT AS frequency
FROM icd_raw
GROUP BY event_year, target_code
""").df().assign(target_system='icd')

# Build CPT aggregated
cpt = con.sql("""
WITH cpt_raw AS (
  SELECT event_year, cpt_mod_1_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE cpt_mod_1_code IS NOT NULL AND cpt_mod_1_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL
  SELECT event_year, cpt_mod_2_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE cpt_mod_2_code IS NOT NULL AND cpt_mod_2_code <> '' AND event_year BETWEEN 2016 AND 2020
)
SELECT event_year::INT AS event_year, target_code, COUNT(*)::BIGINT AS frequency
FROM cpt_raw
GROUP BY event_year, target_code
""").df().assign(target_system='cpt')

all_targets = pd.concat([icd, cpt], ignore_index=True)
con.register('all_targets', all_targets)
con.sql("""
COPY all_targets TO 's3://pgxdatalake/gold/target_code/target_code_latest.parquet'
(FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
""")
con.sql("""
COPY all_targets TO 's3://pgxdatalake/gold/target_code/target_code_latest.csv'
(FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE TRUE)
""")
```

---

## 2) Static Visualizations (Matplotlib/Seaborn helpers)

```python
import sys, pandas as pd
sys.path.append('/home/pgx3874/pgx-analysis')

from helpers_1997_13.visualization_utils import (
    plot_stacked_by_year,
    plot_top_bars,
    save_current_chart,
)

# Load latest from S3 (drug_name)
import duckdb
duckdb.sql("LOAD httpfs; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
drug_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/drug_name/drug_frequency_latest.parquet')").df()

# Stacked by year
plot_stacked_by_year(drug_df, target_col='drug_name', year_col='event_year', freq_col='frequency', title_suffix='Drug Name')
save_current_chart('drugname_stacked', 'drug_name_frequency')

# Top 20
totals = drug_df.groupby('drug_name', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)
plot_top_bars(totals, target_col='drug_name', value_col='frequency', top_n=20, title='Top 20 Drug Names')
save_current_chart('drugname_top20', 'drug_name_frequency')

# Load latest from S3 (target_code)
tc_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')").df()
icd_df = tc_df[tc_df['target_system'] == 'icd']
cpt_df = tc_df[tc_df['target_system'] == 'cpt']
plot_stacked_by_year(icd_df, target_col='target_code', year_col='event_year', freq_col='frequency', title_suffix='ICD')
save_current_chart('icd_stacked', 'target_code_frequency')
plot_stacked_by_year(cpt_df, target_col='target_code', year_col='event_year', freq_col='frequency', title_suffix='CPT')
save_current_chart('cpt_stacked', 'target_code_frequency')
```

---

## 3) Interactive Plotly Dashboards (HTML to S3)

```python
from helpers_1997_13.visualization_utils import create_plotly_frequency_dashboard

# Drug dashboard
create_plotly_frequency_dashboard(
    drug_df,
    title='Drug Frequency Explorer',
    s3_output_path='s3://pgxdatalake/visualizations/drug_name/drug_frequency_dashboard.html',
    target_col='drug_name', year_col='event_year', freq_col='frequency', system_col=None, top_n=20,
)

# Target code dashboard
create_plotly_frequency_dashboard(
    tc_df,
    title='Target Frequency Explorer (ICD/CPT)',
    s3_output_path='s3://pgxdatalake/visualizations/target_code/target_frequency_dashboard.html',
    target_col='target_code', year_col='event_year', freq_col='frequency', system_col='target_system', top_n=20,
)
```

### Optional: Co-Occurrence (multi-select)
For within-domain co-occurrence (drug×drug) or cross-domain (ICD×CPT), compute a `pairs_df` and pass it into the dashboard (the function accepts optional `pairs_df`, `pair_row_col`, `pair_col_col`, `pair_year_col`). Example for drug×drug among selected drugs:

```python
selected_drugs = ['lisinopril', 'metformin', 'amlodipine']
duckdb.sql("CREATE OR REPLACE TEMP TABLE sel AS SELECT * FROM (VALUES {} ) v(drug)".format(
    ",".join([f"('{d.lower()}')" for d in selected_drugs])
))

pairs_df = duckdb.sql("""
WITH base AS (
  SELECT claim_id, event_year::INT AS event_year, LOWER(drug_name) AS drug
  FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band=*/event_year=*/pharmacy_data.parquet')
  WHERE drug_name IS NOT NULL AND drug_name <> '' AND event_year BETWEEN 2016 AND 2020
), sub AS (
  SELECT b.claim_id, b.event_year, b.drug
  FROM base b JOIN sel s ON b.drug = s.drug
)
SELECT
  a.event_year,
  LEAST(a.drug, b.drug) AS drug_a,
  GREATEST(a.drug, b.drug) AS drug_b,
  COUNT(*)::BIGINT AS frequency
FROM sub a JOIN sub b ON a.claim_id=b.claim_id AND a.event_year=b.event_year
WHERE a.drug < b.drug
GROUP BY a.event_year, LEAST(a.drug, b.drug), GREATEST(a.drug, b.drug)
""").df()

# Save latest pairs to S3 if desired
from helpers_1997_13.visualization_utils import write_drug_pairs_latest
write_drug_pairs_latest(pairs_df)

# Dashboard with co-occurrence enabled
create_plotly_frequency_dashboard(
    drug_df,
    title='Drug Frequency Explorer (with Co-Occurrence)',
    s3_output_path='s3://pgxdatalake/visualizations/drug_name/drug_frequency_dashboard.html',
    target_col='drug_name', year_col='event_year', freq_col='frequency', system_col=None, top_n=20,
    pairs_df=pairs_df, pair_row_col='drug_a', pair_col_col='drug_b', pair_year_col='event_year'
)
```

---

## 4) Exporting Filtered Subsets to S3 Parquet

Use DuckDB to export filtered slices:
```python
systems = ('icd','cpt')
targets = ('A10','B20','C30')
ymin, ymax = 2016, 2020

duckdb.sql(f"""
COPY (
  SELECT *
  FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')
  WHERE target_system IN {systems}
    AND target_code IN {targets}
    AND event_year BETWEEN {ymin} AND {ymax}
) TO 's3://pgxdatalake/gold/target_code/exports/filtered_target_codes.parquet'
(FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
""")
```

For `drug_name`, switch the column in the predicate and source to `s3://pgxdatalake/gold/drug_name/drug_frequency_latest.parquet`.



## Notebook Calls by Dataset

### drug_name: Notebook Calls

#### Cell d1: Build backend dataframe and save to S3 (Parquet/CSV)
Option A (preferred): call the existing script functions via importlib
```python
import importlib.util

path = '/home/pgx3874/pgx-analysis/1_apcd_input_data/4_drug_frequency_analysis.py'
spec = importlib.util.spec_from_file_location('drug_freq_module', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = mod.get_drug_frequency_data()
from helpers_1997_13.visualization_utils import write_drug_frequency_latest
write_drug_frequency_latest(df)
```

Option B: inline DuckDB (equivalent)
```python
import duckdb
duckdb.sql("LOAD httpfs; LOAD aws; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
df = duckdb.sql("""
SELECT event_year::INT AS event_year, drug_name, COUNT(*)::BIGINT AS frequency
FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band=*/event_year=*/pharmacy_data.parquet')
WHERE drug_name IS NOT NULL AND drug_name <> '' AND event_year BETWEEN 2016 AND 2020
GROUP BY event_year, drug_name
""").df()
from helpers_1997_13.visualization_utils import write_drug_frequency_latest
write_drug_frequency_latest(df)
```

#### Cell d2: Static visualizations
```python
import duckdb
from helpers_1997_13.visualization_utils import plot_stacked_by_year, plot_top_bars, save_current_chart

duckdb.sql("LOAD httpfs; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
drug_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/drug_name/drug_frequency_latest.parquet')").df()

plot_stacked_by_year(drug_df, target_col='drug_name', year_col='event_year', freq_col='frequency', title_suffix='Drug Name')
save_current_chart('drugname_stacked', 'drug_name_frequency')

totals = drug_df.groupby('drug_name', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)
plot_top_bars(totals, target_col='drug_name', value_col='frequency', top_n=20, title='Top 20 Drug Names')
save_current_chart('drugname_top20', 'drug_name_frequency')
```

#### Cell d3: Interactive dashboard with export
```python
import duckdb
from helpers_1997_13.visualization_utils import create_plotly_frequency_dashboard

duckdb.sql("LOAD httpfs; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
drug_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/drug_name/drug_frequency_latest.parquet')").df()

create_plotly_frequency_dashboard(
    drug_df,
    title='Drug Frequency Explorer',
    s3_output_path='s3://pgxdatalake/visualizations/drug_name/drug_frequency_dashboard.html',
    target_col='drug_name', year_col='event_year', freq_col='frequency', system_col=None, top_n=20,
)
```


### target_code: Notebook Calls

#### Cell t1: Build backend dataframe and save to S3 (Parquet/CSV)
Option A (preferred): run the existing script (writes latest S3 outputs)
```python
import subprocess
subprocess.run([
    '/home/pgx3874/jupyter-env/bin/python3.11',
    '/home/pgx3874/pgx-analysis/1_apcd_input_data/6_target_frequency_analysis.py'
], check=True)
```

Option B: inline DuckDB (equivalent)
```python
import duckdb, pandas as pd
duckdb.sql("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")

icd = duckdb.sql("""
WITH icd_raw AS (
  SELECT event_year, primary_icd_diagnosis_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, two_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE two_icd_diagnosis_code   IS NOT NULL AND two_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, three_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, four_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE four_icd_diagnosis_code  IS NOT NULL AND four_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, five_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE five_icd_diagnosis_code  IS NOT NULL AND five_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, six_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE six_icd_diagnosis_code   IS NOT NULL AND six_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, seven_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE seven_icd_diagnosis_code IS NOT NULL AND seven_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, eight_icd_diagnosis_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE eight_icd_diagnosis_code IS NOT NULL AND eight_icd_diagnosis_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, nine_icd_diagnosis_code  FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE nine_icd_diagnosis_code  IS NOT NULL AND nine_icd_diagnosis_code  <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, ten_icd_diagnosis_code   FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE ten_icd_diagnosis_code   IS NOT NULL AND ten_icd_diagnosis_code   <> '' AND event_year BETWEEN 2016 AND 2020
)
SELECT event_year::INT AS event_year, target_code, COUNT(*)::BIGINT AS frequency
FROM icd_raw GROUP BY event_year, target_code
""").df().assign(target_system='icd')

cpt = duckdb.sql("""
WITH cpt_raw AS (
  SELECT event_year, cpt_mod_1_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE cpt_mod_1_code IS NOT NULL AND cpt_mod_1_code <> '' AND event_year BETWEEN 2016 AND 2020
  UNION ALL SELECT event_year, cpt_mod_2_code AS target_code FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') WHERE cpt_mod_2_code IS NOT NULL AND cpt_mod_2_code <> '' AND event_year BETWEEN 2016 AND 2020
)
SELECT event_year::INT AS event_year, target_code, COUNT(*)::BIGINT AS frequency
FROM cpt_raw GROUP BY event_year, target_code
""").df().assign(target_system='cpt')

import pandas as pd
all_targets = pd.concat([icd, cpt], ignore_index=True)
duckdb.register('all_targets', all_targets)
duckdb.sql("COPY all_targets TO 's3://pgxdatalake/gold/target_code/target_code_latest.parquet' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)")
duckdb.sql("COPY all_targets TO 's3://pgxdatalake/gold/target_code/target_code_latest.csv' (FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE TRUE)")
```

#### Cell t1b: Optional - load precomputed pickle for extended outputs (age-band)
```python
import pickle
pk_path = '1_apcd_input_data/outputs/target_analysis_data.pkl'
with open(pk_path, 'rb') as f:
    tdata = pickle.load(f)

icd_by_pos = tdata['icd_by_position']      # event_year, icd_position, target_code, frequency
icd_agg    = tdata['icd_aggregated']       # event_year, target_code, frequency
icd_by_age = tdata['icd_by_age']           # event_year, target_code, age_band, frequency

cpt_by_fld = tdata['cpt_by_field']         # event_year, cpt_field, target_code, frequency
cpt_agg    = tdata['cpt_aggregated']       # event_year, target_code, frequency
cpt_by_age = tdata['cpt_by_age']           # event_year, target_code, age_band, frequency

all_targets = tdata['all_targets']         # event_year, target_code, frequency, target_system
```

#### Cell t2: Static visualizations
```python
import duckdb
from helpers_1997_13.visualization_utils import plot_stacked_by_year, save_current_chart

duckdb.sql("LOAD httpfs; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
tc_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')").df()

icd_df = tc_df[tc_df['target_system']=='icd']
cpt_df = tc_df[tc_df['target_system']=='cpt']
plot_stacked_by_year(icd_df, target_col='target_code', year_col='event_year', freq_col='frequency', title_suffix='ICD')
save_current_chart('icd_stacked', 'target_code_frequency')
plot_stacked_by_year(cpt_df, target_col='target_code', year_col='event_year', freq_col='frequency', title_suffix='CPT')
save_current_chart('cpt_stacked', 'target_code_frequency')
```

#### Cell t2b: Summary table (overall)
```python
import pandas as pd
tc_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')").df()

# Summary by code
summary = (
    tc_df.groupby(['target_system','target_code'], as_index=False)
         .agg(total_frequency=('frequency','sum'), years_present=('event_year','nunique'))
         .sort_values(['target_system','total_frequency'], ascending=[True, False])
)
summary.head(25)
```

#### Cell t2c: Code-of-interest + similar string matches
```python
code_of_interest = 'F11.20'  # change as needed (ICD or CPT)
needle = code_of_interest.replace('.', '').upper()

tc_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')").df()
tc_df['code_flat'] = tc_df['target_code'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)

# All variants that contain the same flattened substring
similar = (
    tc_df[tc_df['code_flat'].str.contains(needle, na=False)]
      .groupby(['target_system','target_code'], as_index=False)
      .agg(total_frequency=('frequency','sum'), years_present=('event_year','nunique'))
      .sort_values(['target_system','total_frequency'], ascending=[True, False])
)
similar
```

#### Cell t2d: Corrected visual by frequency, by year, by age_band
Option A: using precomputed age-band outputs (from t1b pickle)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Choose domain and subset
domain = 'icd'  # or 'cpt'
df_age = icd_by_age if domain=='icd' else cpt_by_age

# Focus on code of interest and its similar matches
codes = similar['target_code'].unique().tolist()
fdf = df_age[df_age['target_code'].isin(codes)]

plt.figure(figsize=(10,6))
sns.barplot(
    data=fdf,
    x='event_year', y='frequency', hue='age_band',
    estimator=sum, errorbar=None
)
plt.title(f"{domain.upper()} frequency by year and age_band (selected codes)")
plt.xlabel('Year'); plt.ylabel('Frequency')
plt.legend(title='Age Band', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
```

Option B: build age-band on the fly from gold (heavier)
```python
codes = similar['target_code'].unique().tolist()
duckdb.register('codes', pd.DataFrame({'target_code': codes}))
ab = duckdb.sql("""
SELECT m.event_year::INT AS event_year,
       m.member_age_band_dos AS age_band,
       c.target_code,
       COUNT(*)::BIGINT AS frequency
FROM read_parquet('s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet') m
JOIN codes c ON (
     m.primary_icd_diagnosis_code = c.target_code OR m.two_icd_diagnosis_code   = c.target_code OR
     m.three_icd_diagnosis_code   = c.target_code OR m.four_icd_diagnosis_code  = c.target_code OR
     m.five_icd_diagnosis_code    = c.target_code OR m.six_icd_diagnosis_code   = c.target_code OR
     m.seven_icd_diagnosis_code   = c.target_code OR m.eight_icd_diagnosis_code = c.target_code OR
     m.nine_icd_diagnosis_code    = c.target_code OR m.ten_icd_diagnosis_code   = c.target_code
)
WHERE m.member_age_band_dos IS NOT NULL AND m.member_age_band_dos <> ''
  AND m.event_year BETWEEN 2016 AND 2020
GROUP BY 1,2,3
""").df()

plt.figure(figsize=(10,6))
sns.barplot(data=ab, x='event_year', y='frequency', hue='age_band', estimator=sum, errorbar=None)
plt.title('ICD frequency by year and age_band (selected codes)')
plt.tight_layout()
```

#### Cell t3: Interactive dashboard with export
```python
import duckdb
from helpers_1997_13.visualization_utils import create_plotly_frequency_dashboard

duckdb.sql("INSTALL httpfs; LOAD httpfs; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'")
tc_df = duckdb.sql("SELECT * FROM read_parquet('s3://pgxdatalake/gold/target_code/target_code_latest.parquet')").df()

create_plotly_frequency_dashboard(
    tc_df,
    title='Target Frequency Explorer (ICD/CPT)',
    s3_output_path='s3://pgxdatalake/visualizations/target_code/target_frequency_dashboard.html',
    target_col='target_code', year_col='event_year', freq_col='frequency', system_col='target_system', top_n=20,
)
```

---

## 5) QA Correction Workflow (ICD/CPT/Drug codes)

We maintain corrected coding in gold by normalizing known variants. Run the QA script to:
- scan for variants of a given ICD/CPT/drug string,
- produce a proposed mapping (variant -> canonical),
- optionally write corrected outputs.

Example notebook call (script name TBD, e.g., `8_code_correction_qa.py`):
```python
import subprocess
subprocess.run([
    '/home/pgx3874/jupyter-env/bin/python3.11',
    '/home/pgx3874/pgx-analysis/1_apcd_input_data/8_code_correction_qa.py',
    '--domain', 'icd',                    # icd|cpt|drug
    '--needle', 'F11.20',                 # code of interest
    '--apply',                            # include to apply corrections; omit for dry-run report
], check=True)
```
Outputs:
- Report of variants and frequencies
- Proposed canonical form
- If `--apply`, writes corrected gold to new versioned S3 objects (idempotent "latest" keys with S3 versioning)