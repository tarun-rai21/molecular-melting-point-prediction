import pandas as pd
from rdkit import Chem


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for melting point dataset.
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with missing SMILES
    df = df[df["SMILES"].notna()]

    # Remove invalid SMILES
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df[df["mol"].notna()]

    # Fill remaining missing numeric values with 0
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Drop helper mol column (only used for validation)
    df = df.drop(columns=["mol"])

    return df
