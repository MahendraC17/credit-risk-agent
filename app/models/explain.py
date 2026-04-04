import joblib
import pandas as pd
import shap
from app.models.predict import prepare_input
import warnings
warnings.filterwarnings("ignore")


pipeline = joblib.load("app/models/credit_model.pkl")

if hasattr(pipeline, "calibrated_classifiers_"):
    pipeline = pipeline.calibrated_classifiers_[0].estimator

elif hasattr(pipeline, "base_estimator"):
    pipeline = pipeline.base_estimator

elif hasattr(pipeline, "estimators_"):
    pipeline = pipeline.estimators_[0]

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessing"]

explainer = shap.TreeExplainer(model)

def clean_feature_name(feature: str):
    feature = feature.replace("num__", "").replace("cat__", "")

    categorical_fields = {
        "home_ownership",
        "loan_purpose",
        "historical_default"
    }

    for field in categorical_fields:
        if feature.startswith(field + "_"):
            value = feature[len(field) + 1:]
            return field, value

    return feature, None

def explain_prediction(applicant_data: dict, top_n: int = 5):
    df = prepare_input(applicant_data)

    X_transformed = preprocessor.transform(df)

    feature_names = preprocessor.get_feature_names_out()

    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "impact": values
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False)

    cleaned_output = []

    for _, row in shap_df.head(top_n).iterrows():
        raw_feature = row["feature"]
        impact = row["impact"]

        base_feature, category = clean_feature_name(raw_feature)

        effect = (
            "increased model risk score"
            if impact > 0
            else "decreased model risk score"
        )
        
        if category:
            feature_name = f"{base_feature} = {category}"
        else:
            feature_name = base_feature

        cleaned_output.append({
            "feature": feature_name,
            "impact": round(float(impact), 4),
            "effect": effect
        })

    return cleaned_output

if __name__ == "__main__":
    from app.db.queries import fetch_applicant

    applicant = fetch_applicant(3)
    explanation = explain_prediction(applicant)

    print("\nTop Risk Drivers:")
    for item in explanation:
        print(item)