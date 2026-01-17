from typing import Optional
import pandas as pd


def _to_float(code: str) -> Optional[float]:
    
    if pd.isna(code):
        return None
    try:
        return float(str(code))
    except ValueError:
        return None


def map_icd9_to_group(code: str) -> str:
    
    x = _to_float(code)
    if x is None:
        return "Unknown"

    # Circulatory system (390–459, 785)
    if (390 <= x <= 459) or x == 785:
        return "Circulatory"

    # Respiratory system (460–519, 786)
    if (460 <= x <= 519) or x == 786:
        return "Respiratory"

    # Digestive system (520–579, 787)
    if (520 <= x <= 579) or x == 787:
        return "Digestive"

    # Diabetes (250.xx)
    if 250 <= x < 251:
        return "Diabetes"

    # Injury (800–999)
    if 800 <= x <= 999:
        return "Injury"

    # Musculoskeletal (710–739)
    if 710 <= x <= 739:
        return "Musculoskeletal"

    # Genitourinary (580–629)
    if 580 <= x <= 629:
        return "Genitourinary"

    # Neoplasms (140–239)
    if 140 <= x <= 239:
        return "Neoplasms"

    return "Other"


def add_grouped_diagnosis_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        grouped_col = f"{col}_grouped"
        df[grouped_col] = df[col].apply(map_icd9_to_group)
    return df