# BACE1 Dataset

This directory contains a processed dataset for **BACE1 inhibitor prediction** derived from the ChEMBL database.

The dataset was constructed using high-confidence experimental bioactivity measurements and prepared for machine learning experiments.

---

# Data Source

Database: **ChEMBL**

Filtering criteria used to construct the dataset:

- Target: **BACE1 (CHEMBL4822)**
- Assay type: **binding**
- Activity type: **IC50**
- Confidence score: **≥ 8**

Only records with valid SMILES and IC50 measurements were retained.

IC50 values were converted to **pIC50** using:

pIC50 = 9 − log10(IC50 in nM)

---

# Generated Files

## bace1_raw_chembl.csv

Raw activity records downloaded directly from ChEMBL.

Contains experimental measurements before preprocessing.

Typical columns include:

- activity_id
- assay_chembl_id
- assay_description
- canonical_smiles
- standard_value
- standard_units
- standard_relation
- confidence_score
- document_chembl_id

This file may contain duplicate molecules and heterogeneous assay conditions.

---

## bace1_clean_pic50.csv

Cleaned dataset ready for machine learning.

Processing steps applied:

- invalid SMILES removed
- missing activity values removed
- IC50 restricted to **nM units**
- IC50 converted to **pIC50**
- duplicate molecules aggregated using **median activity**

Final columns:

smiles,pIC50

Example:

CCN(CC)CCOC1=CC=CC=C1,7.34  
CCC1=CC=CC=C1,5.12

This file is the **primary dataset used for model training**.

---

## bace1_clean_pic50_scaffold_split.csv

Same dataset as above but with a scaffold-based split.

Additional column:

split

Possible values:

- train
- valid
- test

Example:

smiles,pIC50,split  
CCN(CC)CCOC1=CC=CC=C1,7.34,train  
CCC1=CC=CC=C1,5.12,test

The split is based on **Murcko scaffolds** to prevent scaffold leakage between train and test sets.

---

## bace1_split_indices.json

Stores dataset indices corresponding to each split.

Example structure:

{
  "train": [...],
  "valid": [...],
  "test": [...]
}

This ensures reproducibility across experiments.

---

## dataset_stats.json

Summary statistics describing the dataset.

Typical fields:

- raw_rows
- unique_molecules
- train_size
- valid_size
- test_size
- pic50_mean
- pic50_std
- pic50_min
- pic50_max

This file is useful for quickly inspecting dataset quality and distribution.

---

# Recommended Usage

Typical machine learning workflow:

SMILES  
→ Molecular representation (fingerprint or graph)  
→ Latent representation  
→ Activity predictor (pIC50)

After training, models can be used to screen large compound libraries and identify potential **BACE1 inhibitors**.

---

# Directory Structure
data/
├── bace1_raw_chembl.csv
├── bace1_clean_pic50.csv
├── bace1_clean_pic50_scaffold_split.csv
├── bace1_split_indices.json
└── dataset_stats.json

