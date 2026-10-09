
import streamlit as st

# Configure the dashboard page
st.set_page_config(
    page_title="EY Data Science Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard heading
st.title("EY Data Science Challenge")
st.subheader("Machine Learning Classification Dashboard")

st.write(
    "Explore the dataset, review model performance, "
    "and predict classifications for new records."
)

st.divider()

# Display dataset overview
st.markdown("### Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", "5,899")
col2.metric("Input Features", "7")
col3.metric("Target Classes", "6")
col4.metric("Missing Values", "153")

st.divider()

# Create the two main dashboard tabs
tab1, tab2 = st.tabs([
    "EDA & Model Results",
    "Prediction & Explain"
])


with tab1:
    st.header("EDA & Model Results")

    # Load the original dataset
    import pandas as pd
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "raw" / "Challenge_Data.csv"

    df = pd.read_csv(data_path)

    # Display target class distribution
    st.subheader("Target Class Distribution")

    class_counts = df["ClassificationLabel"].replace({
        "category_1": "Category_1",
        "Category2": "Category_2",
        "Category 3": "Category_3",
        "Category _3": "Category_3",
        "Category4": "Category_4",
        "Category 5": "Category_5",
        "Categry_6": "Category_6"
    }).value_counts()

    st.bar_chart(class_counts)

    # Display data quality information
    st.subheader("Data Quality")

    quality_data = pd.DataFrame({
        "Metric": [
            "Total records",
            "Duplicate records",
            "Missing values in Col4"
        ],
        "Count": [
            len(df),
            int(df.duplicated().sum()),
            int(df["Col4"].isna().sum())
        ]
    })

    st.dataframe(quality_data, hide_index=True, use_container_width=True)

    # Display model comparison results
    st.subheader("Model Comparison")

    model_results = pd.DataFrame({
        "Model": [
            "Tuned Logistic Regression",
            "Random Forest",
            "Linear SVM",
            "XGBoost",
            "Controlled Undersampling"
        ],
        "Accuracy": [0.9541, 0.9492, 0.9557, 0.9344, 0.9393],
        "Balanced Accuracy": [0.8744, 0.8263, 0.7202, 0.7729, 0.7218],
        "Macro F1": [0.7439, 0.6973, 0.6460, 0.6418, 0.6361]
    })

    st.dataframe(
        model_results.style.format({
            "Accuracy": "{:.2%}",
            "Balanced Accuracy": "{:.2%}",
            "Macro F1": "{:.2%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Model metrics are from the notebook's current holdout evaluation."
    )



with tab2:
    st.header("Prediction & Explain")

    st.write(
        "Upload a CSV file containing the seven input features "
        "to generate classification predictions."
    )

    import joblib
    import pandas as pd
    from pathlib import Path

    # Locate the saved model pipeline
    project_dir = Path(__file__).resolve().parent.parent
    model_path = (
        project_dir / "models" / "ey_classification_pipeline.joblib"
    )

    # Define the required input columns
    input_columns = [
        "Col1", "Col2", "Col3", "Col4",
        "Col5", "Col6", "Col7"
    ]

    # Upload the holdout or new dataset
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="prediction_upload"
    )

    if uploaded_file is not None:

        # Read the uploaded file
        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(f"Number of records: {len(uploaded_df):,}")

        # Check that all required features are present
        missing_columns = [
            col for col in input_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        elif uploaded_df.empty:
            st.warning("The uploaded CSV file contains no records.")

        else:
            # Use only the seven required input features
            input_df = uploaded_df[input_columns].copy()

            # Display a preview of the uploaded data
            st.dataframe(
                input_df.head(10),
                use_container_width=True
            )

            # Generate predictions when the user clicks the button
            if st.button("Generate Predictions", type="primary"):

                try:
                    # Load the saved model pipeline
                    model_pipeline = joblib.load(model_path)

                    # Predict the class for each uploaded record
                    predictions = model_pipeline.predict(input_df)

                    # Build the results table
                    results_df = input_df.copy()
                    results_df["Predicted Classification"] = predictions

                    # Add confidence when the model supports probabilities
                    if hasattr(model_pipeline, "predict_proba"):
                        probabilities = model_pipeline.predict_proba(input_df)
                        results_df["Confidence"] = probabilities.max(axis=1)

                    st.success(
                        f"Predictions generated for {len(results_df):,} records."
                    )

                    # Display 25 records per page
                    rows_per_page = 25
                    total_pages = (
                        len(results_df) + rows_per_page - 1
                    ) // rows_per_page

                    page_number = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max(1, total_pages),
                        value=1,
                        step=1,
                        key="prediction_page"
                    )

                    start = (page_number - 1) * rows_per_page
                    end = start + rows_per_page

                    st.dataframe(
                        results_df.iloc[start:end],
                        use_container_width=True
                    )

                    # Download the complete prediction results
                    csv_data = results_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Predictions",
                        data=csv_data,
                        file_name="classification_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

