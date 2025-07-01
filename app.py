import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


def engineer_features(df):
    """
    Engineer features from raw log data for anomaly detection.

    Args:
        df (pd.DataFrame): Raw log data with columns: timestamp, ip_address, http_status, request_path

    Returns:
        pd.DataFrame: DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    df_engineered = df.copy()

    # Convert timestamp column to datetime objects
    df_engineered["timestamp"] = pd.to_datetime(df_engineered["timestamp"])

    # Sort by timestamp to ensure proper time gap calculation
    df_engineered = df_engineered.sort_values("timestamp").reset_index(drop=True)

    # Feature 1: Calculate time gaps between consecutive log entries
    df_engineered["time_gap_seconds"] = (
        df_engineered["timestamp"].diff().dt.total_seconds()
    )
    # Fill the first row (which will be NaN) with 0
    df_engineered["time_gap_seconds"].fillna(0, inplace=True)

    # Feature 2: Calculate IP frequency (how many times each IP appears)
    ip_counts = df_engineered["ip_address"].value_counts()
    df_engineered["ip_frequency"] = df_engineered["ip_address"].map(ip_counts)

    return df_engineered


def highlight_anomalies(row):
    """
    Function to highlight anomalous rows in red.
    """
    if row["anomaly"] == -1:
        return ["background-color: #ffcccc"] * len(row)  # Light red background
    else:
        return [""] * len(row)


# Streamlit App Configuration
st.set_page_config(
    page_title="Log Anomaly Detection Dashboard", page_icon="🔍", layout="wide"
)

# App Title
st.title("🔍 Log Anomaly Detection Dashboard")
st.markdown(
    "Upload your server log CSV file to detect anomalous behavior using machine learning."
)

# Sidebar with information
st.sidebar.header("📋 How it works")
st.sidebar.markdown("""
1. **Upload** your CSV log file
2. **Analyze** logs with Isolation Forest
3. **View** detected anomalies
4. **Examine** anomaly score distribution

**Expected CSV format:**
- timestamp
- ip_address  
- http_status
- request_path
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Details:**")
st.sidebar.markdown("- Algorithm: Isolation Forest")
st.sidebar.markdown("- Contamination: 5%")
st.sidebar.markdown("- Features: Time gaps, IP frequency, HTTP status")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", help="Upload your server log file in CSV format"
)

# Analysis button
analyze_button = st.button("🔍 Analyze Logs", type="primary")

# Main logic - only runs when file is uploaded and button is clicked
if uploaded_file is not None and analyze_button:
    try:
        # Read the uploaded CSV file
        with st.spinner("Loading log data..."):
            df = pd.read_csv(uploaded_file)

        # Display basic info about the dataset
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Log Entries", len(df))
        with col2:
            st.metric("Unique IP Addresses", df["ip_address"].nunique())
        with col3:
            st.metric(
                "Date Range",
                f"{len(pd.to_datetime(df['timestamp']).dt.date.unique())} days",
            )
        with col4:
            st.metric("Unique Status Codes", df["http_status"].nunique())

        # Feature engineering
        with st.spinner("Engineering features..."):
            df_features = engineer_features(df)

        st.success("✅ Feature engineering completed!")

        # Display the first few rows with engineered features
        with st.expander("🔧 View Engineered Features (First 10 rows)"):
            st.dataframe(
                df_features[
                    [
                        "timestamp",
                        "ip_address",
                        "http_status",
                        "time_gap_seconds",
                        "ip_frequency",
                    ]
                ].head(10)
            )

        # Select features for the model
        feature_columns = ["time_gap_seconds", "ip_frequency", "http_status"]
        X = df_features[feature_columns]

        # Train Isolation Forest model
        with st.spinner("Training Isolation Forest model..."):
            # Initialize the model with 5% contamination
            isolation_forest = IsolationForest(
                contamination=0.05, random_state=42, n_estimators=100
            )

            # Fit the model and predict anomalies
            anomaly_predictions = isolation_forest.fit_predict(X)

            # Get anomaly scores
            anomaly_scores = isolation_forest.decision_function(X)

        # Add predictions to the dataframe
        df_features["anomaly"] = anomaly_predictions
        df_features["anomaly_score"] = anomaly_scores

        # Count anomalies
        anomaly_count = (anomaly_predictions == -1).sum()
        normal_count = (anomaly_predictions == 1).sum()

        st.success(
            f"✅ Model training completed! Detected {anomaly_count} anomalies out of {len(df)} log entries."
        )

        # Display results
        st.subheader("🚨 Anomaly Detection Results")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Normal Entries",
                normal_count,
                delta=f"{normal_count / len(df) * 100:.1f}%",
            )
        with col2:
            st.metric(
                "Anomalous Entries",
                anomaly_count,
                delta=f"{anomaly_count / len(df) * 100:.1f}%",
            )
        with col3:
            st.metric("Detection Rate", f"{anomaly_count / len(df) * 100:.1f}%")

        # Display the dataframe with highlighted anomalies
        st.subheader("📋 Log Entries with Anomaly Detection")

        # Create a display version of the dataframe
        display_df = df_features[
            [
                "timestamp",
                "ip_address",
                "http_status",
                "request_path",
                "time_gap_seconds",
                "ip_frequency",
                "anomaly",
                "anomaly_score",
            ]
        ].copy()

        # Replace anomaly values with more readable labels
        display_df["anomaly_label"] = display_df["anomaly"].map(
            {1: "Normal", -1: "Anomaly"}
        )

        # Style the dataframe to highlight anomalies
        styled_df = display_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # Show only anomalies if there are any
        if anomaly_count > 0:
            st.subheader("⚠️ Detected Anomalies")
            anomalies_df = df_features[df_features["anomaly"] == -1][
                [
                    "timestamp",
                    "ip_address",
                    "http_status",
                    "request_path",
                    "time_gap_seconds",
                    "ip_frequency",
                    "anomaly_score",
                ]
            ].copy()
            st.dataframe(anomalies_df, use_container_width=True)

            # Explanation of why entries might be anomalous
            st.markdown("**Possible reasons for anomalies:**")
            st.markdown("- Unusual IP addresses (rare or never seen before)")
            st.markdown("- Large time gaps between consecutive requests")
            st.markdown("- Uncommon HTTP status codes (4xx, 5xx errors)")
            st.markdown("- Combination of multiple unusual factors")

        # Distribution of anomaly scores
        st.subheader("📈 Distribution of Anomaly Scores")
        st.markdown(
            "Negative scores indicate anomalies, positive scores indicate normal behavior."
        )

        # Create histogram data
        score_df = pd.DataFrame(
            {
                "Anomaly Score": anomaly_scores,
                "Label": [
                    "Anomaly" if pred == -1 else "Normal"
                    for pred in anomaly_predictions
                ],
            }
        )

        # Create two columns for the charts
        col1, col2 = st.columns(2)

        with col1:
            # Histogram of all scores
            st.bar_chart(pd.Series(anomaly_scores).value_counts().sort_index())

        with col2:
            # Show statistics
            st.markdown("**Score Statistics:**")
            st.markdown(f"- Mean: {np.mean(anomaly_scores):.3f}")
            st.markdown(f"- Std: {np.std(anomaly_scores):.3f}")
            st.markdown(f"- Min: {np.min(anomaly_scores):.3f}")
            st.markdown(f"- Max: {np.max(anomaly_scores):.3f}")

            # Threshold info
            threshold = np.percentile(
                anomaly_scores, 5
            )  # 5th percentile (contamination level)
            st.markdown(f"- Anomaly Threshold: {threshold:.3f}")

        # Download option for results
        st.subheader("💾 Download Results")

        # Prepare CSV for download
        results_csv = df_features.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=results_csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.markdown(
            "Please ensure your CSV file has the correct format with columns: timestamp, ip_address, http_status, request_path"
        )

elif uploaded_file is not None:
    st.info("👆 Click 'Analyze Logs' button to start the anomaly detection process.")
else:
    st.info("👆 Please upload a CSV file to begin analysis.")

    # Show example of expected format
    st.subheader("📝 Expected CSV Format")
    example_data = {
        "timestamp": [
            "2024-01-15 10:00:00",
            "2024-01-15 10:00:15",
            "2024-01-15 10:00:30",
        ],
        "ip_address": ["192.168.1.10", "10.0.0.5", "192.168.1.12"],
        "http_status": [200, 200, 404],
        "request_path": ["/index.html", "/api/users", "/missing-page"],
    }
    st.dataframe(pd.DataFrame(example_data))

# Footer
st.markdown("---")
