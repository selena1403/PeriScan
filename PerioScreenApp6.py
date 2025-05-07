import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import qrcode
from PIL import Image

# --- Streamlit UI ---
st.set_page_config(page_title="Periodontitis Risk Report", layout="wide")
st.title("🦷 Periodontitis Prediction and SHAP Explanation")

# Increase text size for the title
st.markdown("<h2 style='text-align: center;'>Periodontitis Risk Prediction and Detailed SHAP Explanation</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel file with patient data", type=["xlsx"])

# Load trained model
model = tf.keras.models.load_model("D:/WORK/2. On going/7. New Lap/Research/KSHAP/STUDY_3_DL_routine blood test/XAI_binary_model_ori.h5")

# Define raw and grouped features
Demographic_Clinical_Information = ['sex', 'age', 'bmi', 'pulse', 'sbpL', 'dbpL']
Hematological_Parameters = ['wbc', 'rbc', 'hb', 'hct', 'plt']
Lipid_Profile = ['t_chol', 'hdl', 'ldl']
Oral_health = ['dental_1', 'teeth_3', 'teeth_problem']

feature_groups = {
    "Demographic / Clinical": Demographic_Clinical_Information,
    "Hematological Parameters": Hematological_Parameters,
    "Lipid Profile": Lipid_Profile,
    "Oral Health": Oral_health
}
raw_features = Demographic_Clinical_Information + Hematological_Parameters + Lipid_Profile + Oral_health

# --- Main processing ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "ID" not in df.columns:
        st.error("The uploaded file must contain an 'ID' column.")
    else:
        patient_ids = df["ID"].tolist()
        selected_id = st.selectbox("Select Patient ID", patient_ids)
        selected_patient = df[df["ID"] == selected_id]

        if selected_patient.empty:
            st.warning("Selected patient not found.")
        else:
            # Prepare data
            X_raw = df[raw_features]
            X_encoded = pd.get_dummies(X_raw)
            training_columns = X_encoded.columns
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            input_raw = selected_patient[raw_features]
            input_encoded = pd.get_dummies(input_raw)
            input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)
            input_scaled = scaler.transform(input_encoded)

            # Prediction
            pred_prob = model.predict(input_scaled)[0][0]
            pred_label = "Periodontitis" if pred_prob >= 0.5 else "Non-Periodontitis"

            # Display result with icon
            if pred_label == "Non-Periodontitis":
                st.success("🟢 Prediction: **Non-Periodontitis**")
                st.markdown("<div style='font-size: 18px; color: #CD853F; background-color: #FFFFE0; padding: 10px;'>✅ This patient shows no signs of periodontitis risk based on current biomarkers. Continue with regular dental care.</div>", unsafe_allow_html=True)
            else:
                st.error("🔴 Prediction: **Periodontitis**")
                st.markdown("<div style='font-size: 18px; color: #CD853F; background-color: #FFFFE0; padding: 10px;'>⚠️ This patient may be at risk of periodontitis. Professional dental consultation is recommended for further evaluation.</div>", unsafe_allow_html=True)

            # Conditional styling based on periodontitis risk
            if pred_prob >= 0.5:
                color = "red"
            else:
                color = "green"
            
            # Display the predicted probability with conditional styling
            st.markdown(f"Predicted Probability: <span style='font-size: 18px; color: {color}; padding: 10px;'>{pred_prob:.4f}</span>", unsafe_allow_html=True)

            # SHAP explanation
            st.subheader("🔍 SHAP Force Plots by Feature Group")
            shap.initjs()
            background = shap.sample(X_scaled, 100, random_state=42)
            explainer = shap.Explainer(model, background)
            shap_values = explainer(input_scaled)

            shap_vals = shap_values.values[0]
            input_vals = input_encoded.iloc[0].values
            base_value = shap_values.base_values[0]
            feature_names = input_encoded.columns.tolist()

            for group_name, group_features in feature_groups.items():
                group_indices = [i for i, name in enumerate(feature_names) if name in group_features]
                if not group_indices:
                    continue

                st.markdown(f"<div style='font-size: 20px; font-weight: bold; color: #333;'> {group_name} </div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 2))
                shap.force_plot(
                    base_value,
                    shap_vals[group_indices],
                    input_vals[group_indices],
                    feature_names=[feature_names[i] for i in group_indices],
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                plt.close()

                st.image(buf, caption=f"SHAP Force Plot - {group_name}")
                st.download_button(
                    label=f"⬇️ Download {group_name} SHAP Plot",
                    data=buf,
                    file_name=f"shap_{group_name.replace(' ', '_')}_patient_{selected_id}.png",
                    mime="image/png"
                )

            # QR Code generation for report access
            st.subheader("📲 Access Report via QR Code")
            
            # Update this to your actual report URL or server path
            report_url = f"https://yourappdomain.com/report/{selected_id}"  # <-- Replace this!

            qr = qrcode.make(report_url)
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format='PNG')
            qr_buf.seek(0)

            st.image(qr_buf, caption="Scan to Access the Report")





























