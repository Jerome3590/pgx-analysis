-- Optional: COUNT(DISTINCT mi_person_key) for APCD universe (2016-2019).
-- Verify database/table/column names in AWS Glue before using --athena.
-- Example pattern (adjust to your catalog):
-- SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS n
-- FROM silver_medical.medical_claims
-- WHERE year BETWEEN 2016 AND 2019;

SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS n
FROM silver_medical.medical_claims
WHERE year BETWEEN 2016 AND 2019;
