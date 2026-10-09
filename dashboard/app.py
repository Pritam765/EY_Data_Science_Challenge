
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EY Data Science Dashboard",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Challenge_Data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "ey_classification_pipeline.joblib"

# Ensure Python can import the project's src package.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import the class used when the model pipeline was saved.
from src.feature_preparation import FeaturePreparation  # noqa: F401


INPUT_COLUMNS = [f"Col{i}" for i in range(1, 8)]

LABEL_MAPPING = {
    "category_1": "Category_1",
    "Category2": "Category_2",
    "Category 3": "Category_3",
    "Category _3": "Category_3",
    "Category4": "Category_4",
    "Category 5": "Category_5",
    "Categry_6": "Category_6",
}

# Review threshold is a configurable heuristic, not a validated cutoff.
REVIEW_THRESHOLD = 0.70
PAGE_SIZE = 25


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

@st.cache_data
def load_dataset(file_path, modified_time):
    """Load the dataset used for the EDA dashboard."""
    return pd.read_csv(file_path)


@st.cache_resource
def load_model(file_path, modified_time):
    """Load the saved machine learning pipeline."""
    return joblib.load(file_path)


def explain_prediction(transformed_row, classifier, feature_names, class_index):
    """
    Return the strongest positive feature contributions to a
    Logistic Regression class score.
    """
    if not hasattr(classifier, "coef_"):
        return "Feature contribution explanations are unavailable for this model."

    if hasattr(transformed_row, "toarray"):
        row_values = transformed_row.toarray().ravel()
    else:
        row_values = np.asarray(transformed_row).ravel()

    contributions = row_values * classifier.coef_[class_index]

    # Keep features with positive contributions.
    positive_indices = np.flatnonzero(contributions > 0)

    if len(positive_indices) == 0:
        return "No positive feature contributions were identified."

    # Sort positive contributions from largest to smallest.
    ranked_indices = positive_indices[
        np.argsort(contributions[positive_indices])[::-1]
    ][:3]

    details = [
        f"{feature_names[index]} (+{contributions[index]:.3f})"
        for index in ranked_indices
    ]

    return "Top positive model contributions: " + "; ".join(details)


def generate_predictions(model_pipeline, input_df):
    """Generate predictions, probabilities, and model-based explanations."""
    predictions = model_pipeline.predict(input_df)
    probabilities = model_pipeline.predict_proba(input_df)

    classifier = model_pipeline.named_steps["classifier"]
    feature_prep = model_pipeline.named_steps["feature_preparation"]
    preprocessor = model_pipeline.named_steps["preprocessor"]

    # Prepare and transform features using the same fitted pipeline steps.
    prepared_df = feature_prep.transform(input_df)
    transformed_data = preprocessor.transform(prepared_df)
    feature_names = preprocessor.get_feature_names_out()

    results_df = input_df.copy()
    results_df["Predicted Classification"] = predictions
    results_df["Confidence"] = probabilities.max(axis=1).round(4)
    results_df["Review Required"] = (
        probabilities.max(axis=1) < REVIEW_THRESHOLD
    )

    explanations = []

    for row_index, predicted_class in enumerate(predictions):
        class_index = list(classifier.classes_).index(predicted_class)

        explanation = explain_prediction(
            transformed_row=transformed_data[row_index],
            classifier=classifier,
            feature_names=feature_names,
            class_index=class_index,
        )
        explanations.append(explanation)

    results_df["Model Explanation"] = explanations

    return results_df


# --------------------------------------------------
# 3. Application header
# --------------------------------------------------

st.title("EY Data Science Challenge")
st.subheader("Machine Learning Classification Dashboard")

st.write(
    "Explore the dataset, review model performance, and generate "
    "classification predictions with model-based explanations."
)

st.divider()

# --------------------------------------------------
# 4. Dataset overview
# --------------------------------------------------

st.markdown("### Dataset Overview")

metric_columns = st.columns(4)

metric_columns[0].metric("Original Records", "5,899")
metric_columns[1].metric("Input Features", "7")
metric_columns[2].metric("Target Classes", "6")
metric_columns[3].metric("Missing Values in Col4", "153")

st.divider()

tab1, tab2 = st.tabs(["EDA & Model Results", "Prediction & Explain"])


# --------------------------------------------------
# 5. Tab 1: EDA and model results
# --------------------------------------------------

with tab1:
    st.header("Exploratory Data Analysis")

    if not DATA_PATH.exists():
        st.warning(
            "The challenge dataset is not available in the deployment "
            "environment. Dataset charts cannot be displayed."
        )
    else:
        try:
            df = load_dataset(
                str(DATA_PATH),
                DATA_PATH.stat().st_mtime,
            )

            if "ClassificationLabel" in df.columns:
                df["ClassificationLabel"] = df[
                    "ClassificationLabel"
                ].replace(LABEL_MAPPING)

                st.markdown("#### Target Class Distribution")

                class_counts = df["ClassificationLabel"].value_counts()
                st.bar_chart(class_counts)

                st.markdown("#### Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("#### Data Quality")

                quality_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Total records",
                            "Exact duplicate rows",
                            "Missing values in Col4",
                            "Missing values across all columns",
                        ],
                        "Value": [
                            len(df),
                            int(df.duplicated().sum()),
                            int(df["Col4"].isna().sum())
                            if "Col4" in df.columns else 0,
                            int(df.isna().sum().sum()),
                        ],
                    }
                )

                st.dataframe(
                    quality_df,
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")

    st.divider()
    st.markdown("#### Model Comparison")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Tuned Logistic Regression",
                "Accuracy": 0.9541,
                "Balanced Accuracy": 0.8744,
                "Macro F1": 0.7439,
            },
            {
                "Model": "Random Forest",
                "Accuracy": 0.9492,
                "Balanced Accuracy": 0.8263,
                "Macro F1": 0.6973,
            },
            {
                "Model": "Linear SVM",
                "Accuracy": 0.9557,
                "Balanced Accuracy": 0.7202,
                "Macro F1": 0.6460,
            },
            {
                "Model": "XGBoost",
                "Accuracy": 0.9344,
                "Balanced Accuracy": 0.7729,
                "Macro F1": 0.6418,
            },
            {
                "Model": "Controlled Undersampling",
                "Accuracy": 0.9393,
                "Balanced Accuracy": 0.7218,
                "Macro F1": 0.6361,
            },
        ]
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "These are the previously recorded exploratory holdout results. "
        "Minority-class results are uncertain because some classes have "
        "very few holdout examples."
    )


# --------------------------------------------------
# 6. Tab 2: Prediction and explanation
# --------------------------------------------------

with tab2:
    st.header("Prediction & Explain")

    st.write(
        "Upload a CSV containing Col1 through Col7. The application "
        "will predict the classification and show model-based evidence."
    )

    if not MODEL_PATH.exists():
        st.error(
            "The saved model pipeline is missing. Expected location: "
            f"{MODEL_PATH}"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="prediction_upload",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                missing_columns = [
                    column
                    for column in INPUT_COLUMNS
                    if column not in uploaded_df.columns
                ]

                if missing_columns:
                    st.error(
                        "The uploaded CSV is missing required columns: "
                        + ", ".join(missing_columns)
                    )

                elif uploaded_df.empty:
                    st.warning("The uploaded CSV contains no records.")

                else:
                    input_df = uploaded_df[INPUT_COLUMNS].copy()

                    st.markdown("#### Uploaded Data Preview")
                    st.dataframe(
                        input_df.head(10),
                        use_container_width=True,
                    )

                    st.info(
                        f"Ready to predict {len(input_df):,} records."
                    )

                    if st.button(
                        "Generate Predictions",
                        type="primary",
                    ):
                        try:
                            model_pipeline = load_model(
                                str(MODEL_PATH),
                                MODEL_PATH.stat().st_mtime,
                            )

                            with st.spinner(
                                "Generating predictions and explanations..."
                            ):
                                results_df = generate_predictions(
                                    model_pipeline,
                                    input_df,
                                )

                            st.session_state["prediction_results"] = (
                                results_df
                            )

                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

            except Exception as exc:
                st.error(f"Could not read the uploaded CSV: {exc}")

        # Show results retained in session state.
        if "prediction_results" in st.session_state:
            results_df = st.session_state["prediction_results"]

            st.success(
                f"Predictions generated for {len(results_df):,} records."
            )

            review_count = int(results_df["Review Required"].sum())

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Records Predicted", f"{len(results_df):,}")
            metric2.metric(
                "Records Flagged for Review",
                f"{review_count:,}",
            )
            metric3.metric(
                "Mean Maximum Class Probability",
                f"{results_df['Confidence'].mean():.1%}",
            )

            st.caption(
                f"Review flag: maximum predicted class probability below "
                f"{REVIEW_THRESHOLD:.0%}. This is a heuristic and has not "
                "been validated as a calibrated confidence threshold."
            )

            st.markdown("#### Prediction Results")

            total_pages = max(
                1,
                (len(results_df) + PAGE_SIZE - 1) // PAGE_SIZE,
            )

            page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )

            start_index = (page_number - 1) * PAGE_SIZE
            end_index = min(
                start_index + PAGE_SIZE,
                len(results_df),
            )

            st.dataframe(
                results_df.iloc[start_index:end_index],
                use_container_width=True,
                hide_index=False,
            )

            st.caption(
                f"Showing records {start_index + 1:,}–{end_index:,} "
                f"of {len(results_df):,}."
            )

            st.markdown("#### Explain an Individual Prediction")

            selected_row = st.number_input(
                "Record number",
                min_value=1,
                max_value=len(results_df),
                value=1,
                step=1,
            )

            selected_record = results_df.iloc[selected_row - 1]

            st.write(
                "**Predicted classification:**",
                selected_record["Predicted Classification"],
            )
            st.write(
                "**Maximum class probability:**",
                f"{selected_record['Confidence']:.1%}",
            )
            st.write(
                "**Review required:**",
                "Yes" if selected_record["Review Required"] else "No",
            )
            st.write("**Model evidence:**")
            st.write(selected_record["Model Explanation"])

            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions and Explanations",
                data=csv_data,
                file_name="classification_predictions.csv",
                mime="text/csv",
            )
