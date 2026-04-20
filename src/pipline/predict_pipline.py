import os
import sys
from pathlib import Path
import pandas as pd

from typing import Annotated
from pydantic import BaseModel, field_validator
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def predict(self, features):
        try:
            project_root = Path(__file__).resolve().parents[2]
            artifacts_dir = project_root / "artifacts"
            model_path = artifacts_dir / "model.pkl"
            preprocessor_path = artifacts_dir / "preprocessor.pkl"

            # fallback fix
            if not preprocessor_path.exists():
                preprocessor_path = artifacts_dir / "proprocessor.pkl"

            model = load_object(str(model_path))
            preprocessor = load_object(str(preprocessor_path))

            # safer encoder handling
            try:
                encoder = (
                    preprocessor
                    .named_transformers_["cat_pipelines"]
                    .named_steps["one_hot_encoder"]
                )
                if isinstance(encoder, OneHotEncoder):
                    encoder.handle_unknown = "ignore"
            except Exception:
                pass

            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# 🔹 Clean Input Schema
# -----------------------------

def normalize_text(v: str) -> str:
    return v.strip().lower()


def normalize_parent_edu(v: str) -> str:
    v = normalize_text(v)
    return {
        "some collage": "some college",
        "bachelor": "bachelor's degree",
        "masters": "master's degree",
    }.get(v, v)


def normalize_lunch(v: str) -> str:
    v = normalize_text(v)
    return {
        "standrad": "standard",
        "std": "standard",
    }.get(v, v)


class CustomData(BaseModel):
    gender: Annotated[str, normalize_text]
    race_ethnicity: Annotated[str, normalize_text]
    parental_level_of_education: Annotated[str, normalize_parent_edu]
    lunch: Annotated[str, normalize_lunch]
    test_preparation_course: Annotated[str, normalize_text]

    reading_score: int
    writing_score: int

    # fallback validator (ensures Annotated always applied)
    @field_validator("*", mode="before")
    @classmethod
    def apply_normalization(cls, v, info):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    def to_dataframe(self):
        return pd.DataFrame({
            "gender": [self.gender],
            "race/ethnicity": [self.race_ethnicity],
            "parental level of education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test preparation course": [self.test_preparation_course],
            "reading score": [self.reading_score],
            "writing score": [self.writing_score],
        })