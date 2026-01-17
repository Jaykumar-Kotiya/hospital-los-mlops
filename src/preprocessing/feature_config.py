# Target column: this is our Length of Stay (LOS) in days
TARGET_COLUMN = "time_in_hospital"

# Numeric features: continuous/count variables
NUMERIC_FEATURES = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Demographic + admission categorical features
DEMOGRAPHIC_CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
]

ADMISSION_CATEGORICAL_FEATURES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Diagnosis columns (raw ICD-9 codes from dataset)
DIAGNOSIS_FEATURES = ["diag_1", "diag_2", "diag_3"]

# Medication-related features (categorical)
MEDICATION_FEATURES = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# All columns we care about (except target)
BASE_FEATURES = (
    DEMOGRAPHIC_CATEGORICAL_FEATURES
    + ADMISSION_CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + DIAGNOSIS_FEATURES
    + MEDICATION_FEATURES
)