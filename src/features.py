import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ==============================
# Morgan Fingerprint Transformer
# ==============================
class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col="SMILES", radius=2, n_bits=2048):
        self.smiles_col = smiles_col
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        self.fp_generator = GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits
        )
        return self

    def transform(self, X):
        mols = X[self.smiles_col].apply(Chem.MolFromSmiles)
        fps = mols.apply(lambda mol: self.fp_generator.GetFingerprint(mol))
        fp_array = np.array([list(fp) for fp in fps])

        return pd.DataFrame(
            fp_array,
            index=X.index,
            columns=[f"fp_{i}" for i in range(self.n_bits)]
        )


# ==============================
# RDKit Descriptor Transformer
# ==============================
class RDKitDescriptorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_col="SMILES"):
        self.smiles_col = smiles_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mols = X[self.smiles_col].apply(Chem.MolFromSmiles)

        desc = mols.apply(lambda mol: {
            "MolWt": Descriptors.MolWt(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumDonors": Descriptors.NumHDonors(mol),
            "NumAcceptors": Descriptors.NumHAcceptors(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "MolMR": Descriptors.MolMR(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "RingCount": Descriptors.RingCount(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "LabuteASA": Descriptors.LabuteASA(mol),
            "BalabanJ": Descriptors.BalabanJ(mol),
        })

        return pd.DataFrame(desc.tolist(), index=X.index)


# ==============================
# Feature Union Builder
# ==============================
def build_feature_union(smiles_col="SMILES"):
    return ColumnTransformer(
        transformers=[
            ("fingerprints", MorganFingerprintTransformer(smiles_col=smiles_col), [smiles_col]),
            ("rdkit", RDKitDescriptorTransformer(smiles_col=smiles_col), [smiles_col]),
        ],
        remainder="drop"
    )
