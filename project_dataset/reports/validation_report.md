# Validation Report

**Generated:** 2026-08-24T17:22:49.723972

## Count Validation
- Human: 5000 (expected 5000) [OK]
- AI: 5000 (expected 5000) [OK]
- Total: 10000 (expected 10000) [OK]

## Label Validation
- Human records all have label 0 [OK]
- AI records all have label 1 [OK]

## Uniqueness Validation
- Sample IDs unique: [OK]
- Source row IDs unique (human): [OK]
- Source row IDs unique (AI): [OK]
- Texts unique (human): [OK]
- Texts unique (AI): [OK]
- Cross-set duplicates: 0 [OK]

## Provenance Validation
- All required fields present: [OK]
- No nulls in required fields: [OK]

## Reproducibility Test
- Seed: 2026
- Human source IDs match: [OK]
- AI source IDs match: [OK]
- Overall: PASSED

## Output Files
- Parquet: project_dataset\sampled\originals_10000.parquet
- CSV: project_dataset\exports\originals_10000.csv
