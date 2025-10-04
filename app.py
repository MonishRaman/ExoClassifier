import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load saved model and scaler
model = joblib.load("exoplanet_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="NASA Exoplanet Classifier", layout="wide")
st.title("🪐 NASA Exoplanet Classification Dashboard")
st.markdown("Upload data or enter parameters to predict exoplanet status and explore them interactively.")

# Sidebar: input options
st.sidebar.header("📂 Input Options")
input_choice = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Entry"])

# Expected columns
columns = [
    "koi_period", "koi_duration", "koi_depth", "koi_ror", "koi_prad", "koi_sma",
    "koi_incl", "koi_teq", "koi_insol", "koi_eccen", "koi_impact", "koi_num_transits",
    "koi_model_snr", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_kepmag",
    "koi_gmag", "koi_rmag", "koi_jmag"
]

st.sidebar.markdown("**Expected CSV columns:**")
st.sidebar.code(", ".join(columns), language="text")

# Initialize empty DataFrame
data = pd.DataFrame()

# --- Option 1: Upload CSV ---
if input_choice == "Upload CSV":
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        try:
            data = pd.read_csv(file, engine="python", on_bad_lines="skip")
            st.write("### Preview of Uploaded Data")
            st.dataframe(data.head())

            # Validate columns
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded data: {missing_cols}")
            else:
                # Handle NaNs by imputing mean
                data[columns] = data[columns].fillna(data[columns].mean())

                # Scale and predict
                scaled = scaler.transform(data[columns])
                preds = model.predict(scaled)
                data["Prediction"] = preds
                data["Prediction"] = data["Prediction"].map({
                    0: "FALSE POSITIVE",
                    1: "CANDIDATE",
                    2: "CONFIRMED"
                })

                st.success("✅ Prediction Complete!")
                st.dataframe(data[["Prediction"] + columns].head(10))

                # Download results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")

# --- Option 2: Manual Entry ---
if input_choice == "Manual Entry":
    st.subheader("Enter Exoplanet Parameters Manually")
    values = []
    for c in columns:
        val = st.number_input(f"{c}", value=0.0)
        values.append(val)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([values], columns=columns)
            # Handle NaNs if any
            input_df[columns] = input_df[columns].fillna(input_df[columns].mean())
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]

            label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
            st.success(f"🌍 Prediction: **{label_map[pred]}**")

            # Add manual entry to data for plotting
            input_df["Prediction"] = label_map[pred]
            data = pd.concat([data, input_df], ignore_index=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --- Summary Statistics ---
if not data.empty:
    st.subheader("📊 Summary Statistics")
    summary = data.groupby("Prediction")[columns].mean().reset_index()
    st.dataframe(summary)

    counts = data["Prediction"].value_counts().reset_index()
    counts.columns = ["Prediction", "Count"]
    st.bar_chart(counts.set_index("Prediction"))

# --- 3D Plot Explorer ---
if not data.empty:
    st.subheader("🌌 Interactive 3D Scatter Plot Explorer")

    # Filter panel
    st.sidebar.subheader("📊 Filters")
    pred_options = st.sidebar.multiselect("Filter by Prediction:", options=data["Prediction"].unique(), default=data["Prediction"].unique())
    teq_min, teq_max = st.sidebar.slider("Filter by Equilibrium Temperature (koi_teq):", int(data["koi_teq"].min()), int(data["koi_teq"].max()), (int(data["koi_teq"].min()), int(data["koi_teq"].max())))
    prad_min, prad_max = st.sidebar.slider("Filter by Planetary Radius (koi_prad):", float(data["koi_prad"].min()), float(data["koi_prad"].max()), (float(data["koi_prad"].min()), float(data["koi_prad"].max())))

    # Apply filters
    filtered_data = data[
        (data["Prediction"].isin(pred_options)) &
        (data["koi_teq"] >= teq_min) & (data["koi_teq"] <= teq_max) &
        (data["koi_prad"] >= prad_min) & (data["koi_prad"] <= prad_max)
    ]

    numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Dropdowns for axes, color, and size
    x_axis = st.selectbox("Select X-axis:", numeric_cols, index=numeric_cols.index("koi_prad"))
    y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=numeric_cols.index("koi_teq"))
    z_axis = st.selectbox("Select Z-axis:", numeric_cols, index=numeric_cols.index("koi_period"))
    color_axis = st.selectbox("Color points by:", numeric_cols + ["Prediction"], index=numeric_cols.index("koi_steff"))
    size_axis = st.selectbox("Size points by:", numeric_cols, index=numeric_cols.index("koi_prad"))

    # 3D scatter plot
    fig = px.scatter_3d(
        filtered_data,
        x=x_axis, y=y_axis, z=z_axis,
        color=color_axis,
        size=size_axis,
        hover_data=["Prediction"] + numeric_cols,
        color_continuous_scale=px.colors.sequential.Viridis if color_axis != "Prediction" else px.colors.qualitative.Set2
    )

    st.plotly_chart(fig, use_container_width=True)
