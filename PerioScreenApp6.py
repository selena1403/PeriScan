import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

# --- Streamlit UI ---
st.set_page_config(page_title="Periodontitis Risk Report", layout="wide")
st.title("🦷 Periodontitis Prediction and SHAP Explanation")
st.markdown("<h2 style='text-align: center;'>Periodontitis Risk Prediction and Detailed SHAP Explanation</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel file with patient data", type=["xlsx"])
model = tf.keras.models.load_model("XAI_binary_model_ori.h5")

# --- Define Features ---
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

        if not selected_patient.empty:
            # Prepare data
            X_raw = df[raw_features]
            X_encoded = pd.get_dummies(X_raw)
            training_columns = X_encoded.columns
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            input_raw = selected_patient[raw_features]
            input_encoded = pd.get_dummies(input_raw).reindex(columns=training_columns, fill_value=0)
            input_scaled = scaler.transform(input_encoded)

            # Prediction
            pred_prob = model.predict(input_scaled)[0][0]
            pred_label = "Periodontitis" if pred_prob >= 0.5 else "Non-Periodontitis"
            color = "red" if pred_prob >= 0.5 else "green"

            # Display prediction
            st.markdown(f"## 🧪 Prediction: <span style='color: {color};'>{pred_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Predicted Probability:** <span style='font-size:18px; color:{color};'>{pred_prob:.4f}</span>", unsafe_allow_html=True)

            # SHAP values
            shap.initjs()
            background = shap.sample(X_scaled, 100, random_state=42)
            explainer = shap.Explainer(model, background)
            shap_values = explainer(input_scaled)

            shap_vals = shap_values.values[0]
            input_vals = input_encoded.iloc[0].values
            base_value = shap_values.base_values[0]
            feature_names = input_encoded.columns.tolist()

            image_paths = []
            for group_name, group_features in feature_groups.items():
                group_indices = [i for i, name in enumerate(feature_names) if name in group_features]
                if not group_indices:
                    continue

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
                img_path = f"shap_{group_name.replace(' ', '_')}_patient_{selected_id}.png"
                with open(img_path, "wb") as f:
                    f.write(buf.read())
                image_paths.append((group_name, img_path))
                st.image(img_path, caption=f"SHAP Force Plot - {group_name}")

            # --- Generate Final Summary Report PNG ---
            font_title = ImageFont.truetype("arial.ttf", 48) if os.name == "nt" else ImageFont.load_default()
            font_body = ImageFont.truetype("arial.ttf", 28) if os.name == "nt" else ImageFont.load_default()

            report_width = 1400
            line_height = 70
            padding = 50
            report_height = padding + (line_height * 4) + len(image_paths) * 300 + 300
            report_img = Image.new("RGB", (report_width, report_height), "white")
            draw = ImageDraw.Draw(report_img)

            draw.text((padding, padding), f"Periodontitis Prediction Report", fill="black", font=font_title)
            draw.text((padding, padding + line_height), f"Patient ID: {selected_id}", fill="black", font=font_body)
            draw.text((padding, padding + line_height*2), f"Prediction: {pred_label}", fill=color, font=font_body)
            draw.text((padding, padding + line_height*3), f"Probability: {pred_prob:.4f}", fill="black", font=font_body)

            y_offset = padding + line_height * 4
            for group_name, path in image_paths:
                shap_img = Image.open(path).resize((1200, 200))
                report_img.paste(shap_img, (padding, y_offset))
                y_offset += shap_img.size[1] + 40

            summary_path = f"Periodontitis_Report_{selected_id}.png"
            report_img.save(summary_path)

            # QR Code
            report_url = f"https://your_hosting_url/{summary_path}"  # Update to real host
            qr = qrcode.make(report_url)
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format='PNG')
            qr_buf.seek(0)

            st.subheader("📝 Full Summary Report")
            st.image(summary_path, caption="Complete Prediction Report")
            st.download_button("⬇️ Download Full Report as PNG", data=open(summary_path, "rb"), file_name=summary_path, mime="image/png")

            st.image(qr_buf, caption="📲 Scan QR to Access Report")
        else:
            st.warning("Selected patient not found.")



















