

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EY Data Science Challenge",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Challenge_Data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "ey_classification_pipeline.joblib"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Required when loading the saved pipeline.
from src.feature_preparation import FeaturePreparation  # noqa: F401
from src.explanation_agent import explain_prediction


INPUT_COLUMNS = [f"Col{i}" for i in range(1, 8)]
REVIEW_THRESHOLD = 0.70
PAGE_SIZE = 25

LABEL_MAPPING = {
    "category_1": "Category_1",
    "Category2": "Category_2",
    "Category 3": "Category_3",
    "Category _3": "Category_3",
    "Category4": "Category_4",
    "Category 5": "Category_5",
    "Categry_6": "Category_6",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_dataset(file_path, modified_time):
    """Load the dataset used for exploratory analysis."""
    return pd.read_csv(file_path)


@st.cache_resource
def load_model(file_path, modified_time):
    """Load the trained classification pipeline."""
    return joblib.load(file_path)


def get_feature_contributions(
    transformed_row,
    classifier,
    feature_names,
    class_index,
    top_n=3,
):
    """Return the strongest positive linear-model contributions."""

    if not hasattr(classifier, "coef_"):
        return []

    if hasattr(transformed_row, "toarray"):
        values = transformed_row.toarray().ravel()
    else:
        values = np.asarray(transformed_row).ravel()

    contributions = values * classifier.coef_[class_index]
    positive_indices = np.flatnonzero(contributions > 0)

    if len(positive_indices) == 0:
        return []

    ranked_indices = positive_indices[
        np.argsort(contributions[positive_indices])[::-1]
    ][:top_n]

    return [
        f"{feature_names[i]} (+{contributions[i]:.3f})"
        for i in ranked_indices
    ]


def generate_predictions(model_pipeline, input_df):
    """Generate classifications, probabilities and model evidence."""

    predictions = model_pipeline.predict(input_df)
    probabilities = model_pipeline.predict_proba(input_df)

    feature_prep = model_pipeline.named_steps["feature_preparation"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    prepared_df = feature_prep.transform(input_df)
    transformed_data = preprocessor.transform(prepared_df)
    feature_names = preprocessor.get_feature_names_out()

    max_probabilities = probabilities.max(axis=1)

    results_df = input_df.copy()
    results_df["Predicted Classification"] = predictions
    results_df["Confidence"] = max_probabilities.round(4)
    results_df["Review Required"] = (
        max_probabilities < REVIEW_THRESHOLD
    )

    evidence_list = []
    explanation_list = []

    for i, predicted_class in enumerate(predictions):
        class_index = list(classifier.classes_).index(predicted_class)

        evidence = get_feature_contributions(
            transformed_row=transformed_data[i],
            classifier=classifier,
            feature_names=feature_names,
            class_index=class_index,
        )

        evidence_list.append(evidence)

        if evidence:
            explanation_list.append(
                "Top positive model contributions: "
                + "; ".join(evidence)
            )
        else:
            explanation_list.append(
                "No positive feature contributions identified."
            )

    results_df["Model Evidence"] = [
        "; ".join(items) if items else "No positive contributions identified."
        for items in evidence_list
    ]
    results_df["Model Explanation"] = explanation_list

    return results_df, evidence_list


# ============================================================
# 3. HEADER
# ============================================================

st.title("EY Data Science Challenge")
st.subheader("Machine Learning Classification Dashboard")

st.write(
    "Explore the dataset, compare classification models, generate "
    "predictions, and review model-based and LLM-generated explanations."
)

st.divider()

metric_columns = st.columns(4)
metric_columns[0].metric("Original Records", "5,899")
metric_columns[1].metric("Input Features", "7")
metric_columns[2].metric("Target Classes", "6")
metric_columns[3].metric("Missing Values in Col4", "153")

tab1, tab2 = st.tabs(
    ["EDA & Model Results", "Prediction & Explain"]
)


# ============================================================
# 4. TAB 1: EDA AND MODEL RESULTS
# ============================================================

with tab1:
    st.header("Exploratory Data Analysis")

    if not DATA_PATH.exists():
        st.warning(
            "The original dataset is not available in this deployment. "
            "Upload data in the prediction tab to use the classifier."
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

                st.markdown("### Target Class Distribution")
                st.bar_chart(
                    df["ClassificationLabel"].value_counts()
                )

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

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
                        if "Col4" in df.columns
                        else 0,
                        int(df.isna().sum().sum()),
                    ],
                }
            )

            st.markdown("### Data Quality")
            st.dataframe(
                quality_df,
                hide_index=True,
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")

    st.divider()
    st.markdown("### Model Comparison")

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
        "Previously recorded exploratory holdout results. Minority "
        "classes have very few examples, so their performance estimates "
        "may be unstable. These metrics are not recomputed by this app."
    )


# ============================================================
# 5. TAB 2: PREDICTION AND EXPLANATION
# ============================================================

with tab2:
    st.header("Prediction & Explain")

    st.write(
        "Upload a CSV containing Col1 through Col7. The app will "
        "generate classifications, model evidence and optional "
        "Ollama-generated explanations."
    )

    if not MODEL_PATH.exists():
        st.error(
            "The saved model pipeline was not found at: "
            f"{MODEL_PATH}"
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload input CSV",
            type=["csv"],
            key="prediction_upload",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                missing_columns = [
                    col for col in INPUT_COLUMNS
                    if col not in uploaded_df.columns
                ]

                if missing_columns:
                    st.error(
                        "The CSV is missing these required columns: "
                        + ", ".join(missing_columns)
                    )

                elif uploaded_df.empty:
                    st.warning("The uploaded CSV contains no records.")

                else:
                    input_df = uploaded_df[INPUT_COLUMNS].copy()

                    st.markdown("### Uploaded Data Preview")
                    st.dataframe(
                        input_df.head(10),
                        use_container_width=True,
                    )

                    st.info(
                        f"{len(input_df):,} records are ready for prediction."
                    )

                    if st.button(
                        "Generate Predictions",
                        type="primary",
                    ):
                        try:
                            pipeline = load_model(
                                str(MODEL_PATH),
                                MODEL_PATH.stat().st_mtime,
                            )

                            with st.spinner(
                                "Generating predictions..."
                            ):
                                results_df, evidence_list = (
                                    generate_predictions(
                                        pipeline,
                                        input_df,
                                    )
                                )

                            st.session_state["prediction_results"] = (
                                results_df
                            )
                            st.session_state["prediction_evidence"] = (
                                evidence_list
                            )
                            st.session_state.pop(
                                "ai_explanation",
                                None,
                            )
                            st.session_state.pop(
                                "ai_explanation_source",
                                None,
                            )
                            st.session_state.pop(
                                "ai_explanation_row",
                                None,
                            )

                        except Exception as exc:
                            st.error(
                                f"Prediction generation failed: {exc}"
                            )

            except Exception as exc:
                st.error(f"Could not read the uploaded CSV: {exc}")

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if "prediction_results" in st.session_state:
            results_df = st.session_state["prediction_results"]
            evidence_list = st.session_state["prediction_evidence"]

            st.success(
                f"Predictions generated for {len(results_df):,} records."
            )

            review_count = int(results_df["Review Required"].sum())
            mean_probability = results_df["Confidence"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Records Predicted",
                f"{len(results_df):,}",
            )
            col2.metric(
                "Flagged for Review",
                f"{review_count:,}",
            )
            col3.metric(
                "Mean Maximum Class Probability",
                f"{mean_probability:.1%}",
            )

            st.caption(
                f"Review Required is set when the maximum predicted "
                f"class probability is below {REVIEW_THRESHOLD:.0%}. "
                "This is a heuristic, not a validated confidence threshold."
            )

            st.markdown("### Prediction Results")

            total_pages = max(
                1,
                (len(results_df) + PAGE_SIZE - 1) // PAGE_SIZE,
            )

            page_number = st.number_input(
                "Page number",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="prediction_page",
            )

            start_index = (int(page_number) - 1) * PAGE_SIZE
            end_index = min(
                start_index + PAGE_SIZE,
                len(results_df),
            )

            st.dataframe(
                results_df.iloc[start_index:end_index],
                use_container_width=True,
            )

            st.caption(
                f"Showing records {start_index + 1:,}–{end_index:,} "
                f"of {len(results_df):,}."
            )

            # ------------------------------------------------
            # INDIVIDUAL PREDICTION
            # ------------------------------------------------

            st.divider()
            st.markdown("### Explain an Individual Prediction")

            selected_row = st.number_input(
                "Record number to explain",
                min_value=1,
                max_value=len(results_df),
                value=1,
                step=1,
                key="selected_prediction_row",
            )

            row_index = int(selected_row) - 1
            selected_record = results_df.iloc[row_index]
            selected_evidence = evidence_list[row_index]

            st.markdown("#### Prediction Summary")

            summary_col1, summary_col2 = st.columns(2)
            summary_col1.metric(
                "Predicted Classification",
                str(selected_record["Predicted Classification"]),
            )
            summary_col2.metric(
                "Maximum Class Probability",
                f"{selected_record['Confidence']:.1%}",
            )

            if selected_record["Review Required"]:
                st.warning("This record is flagged for review.")
            else:
                st.info("This record is not flagged by the review threshold.")

            st.markdown("#### Model-Based Evidence")
            st.write(selected_record["Model Explanation"])

            # -----------------------------------------------
            # OLLAMA LLM EXPLANATION
            # -----------------------------------------------

            st.markdown("### AI-Generated Explanation")

            if st.button(
                "Generate AI Explanation",
                key="generate_ai_explanation",
            ):
                try:
                    with st.spinner(
                        "Generating explanation with local Ollama..."
                    ):
                        ai_result = explain_prediction(
                            predicted_class=str(
                                selected_record[
                                    "Predicted Classification"
                                ]
                            ),
                            probability=float(
                                selected_record["Confidence"]
                            ),
                            feature_contributions=selected_evidence,
                        )

                    if (
                        isinstance(ai_result, tuple)
                        and len(ai_result) == 2
                    ):
                        ai_text, source = ai_result
                    else:
                        ai_text = str(ai_result)
                        source = "Unverified response"

                    st.session_state["ai_explanation"] = ai_text
                    st.session_state["ai_explanation_source"] = source
                    st.session_state["ai_explanation_row"] = row_index

                except Exception as exc:
                    st.error(
                        f"Could not generate an explanation: {exc}"
                    )

            # Display the explanation for the selected record only.
            if (
                "ai_explanation" in st.session_state
                and st.session_state.get("ai_explanation_row") == row_index
            ):
                source = st.session_state.get(
                    "ai_explanation_source",
                    "Unverified response",
                )

                if source == "Ollama LLM":
                    st.success(
                        "Explanation generated successfully by local Ollama LLM."
                    )
                elif source == "Hugging Face LLM":
                    st.success(
                        "Explanation generated successfully by Hugging Face LLM."
                    )
                elif source == "Fallback":
                    st.warning(
                        "LLM unavailable; displaying a fallback response. "
                        "Check the VS Code terminal for the error."
                    )
                else:
                    st.info(
                        "The explanation source has not been confirmed."
                    )

                st.write(st.session_state["ai_explanation"])

            # -----------------------------------------------
            # DOWNLOAD RESULTS
            # -----------------------------------------------

            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions and Explanations",
                data=csv_data,
                file_name="classification_predictions.csv",
                mime="text/csv",
            )
