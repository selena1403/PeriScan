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
st.title("\U0001F9B7 Periodontitis Prediction and SHAP Explanation")
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

# Friendly feature names for output
feature_name_map = {
    'sex': 'Sex', 'age': 'Age', 'bmi': 'BMI', 'pulse': 'Pulse', 'sbpL': 'Systolic BP', 'dbpL': 'Diastolic BP',
    'wbc': 'White Blood Cells', 'rbc': 'Red Blood Cells', 'hb': 'Hemoglobin', 'hct': 'Hematocrit', 'plt': 'Platelets',
    't_chol': 'Total Cholesterol', 'hdl': 'HDL', 'ldl': 'LDL',
    'dental_1': 'Chewing Discomfort Score', 'teeth_3': 'Remaining Teeth Count', 'teeth_problem': 'Problematic Teeth Count'
}

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
            X_raw = df[raw_features]
            X_encoded = pd.get_dummies(X_raw)
            training_columns = X_encoded.columns
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            input_raw = selected_patient[raw_features]
            input_encoded = pd.get_dummies(input_raw).reindex(columns=training_columns, fill_value=0)
            input_scaled = scaler.transform(input_encoded)

            pred_prob = model.predict(input_scaled)[0][0]
            pred_label = "Periodontitis" if pred_prob >= 0.5 else "Non-Periodontitis"

            if pred_label == "Non-Periodontitis":
                st.success("🟢 Prediction: **Non-Periodontitis**")
                st.markdown("<div style='font-size: 18px; color: #CD853F; background-color: #FFFFE0; padding: 10px;'>✅ This patient shows no signs of periodontitis risk based on current biomarkers. Continue with regular dental care.</div>", unsafe_allow_html=True)
            else:
                st.error("🔴 Prediction: **Periodontitis**")
                st.markdown("<div style='font-size: 18px; color: #CD853F; background-color: #FFFFE0; padding: 10px;'>⚠️ This patient may be at risk of periodontitis. Professional dental consultation is recommended for further evaluation.</div>", unsafe_allow_html=True)

            color = "red" if pred_prob >= 0.5 else "green"
            st.markdown(f"Predicted Probability: <span style='font-size: 18px; color: {color}; padding: 10px;'>{pred_prob:.4f}</span>", unsafe_allow_html=True)

           # SHAP explanation setup
            background = shap.sample(X_scaled, 100, random_state=42)
            explainer = shap.Explainer(model, background)
            shap_values = explainer(input_scaled)
            shap_vals = shap_values.values[0]
            input_vals = input_encoded.iloc[0].values
            base_value = shap_values.base_values[0]
            feature_names = input_encoded.columns.tolist()
            
            # SHAP bar chart
            friendly_feature_names = [feature_name_map.get(name, name) for name in feature_names]
            shap_df = pd.DataFrame({
                "Feature": friendly_feature_names,
                "Value": input_vals,
                "SHAP": shap_vals
            })
            shap_df["Abs_SHAP"] = shap_df["SHAP"].abs()
            shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)
            shap_df["Color"] = shap_df["SHAP"].apply(lambda x: "red" if x > 0 else "green")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=shap_df["Color"])
            ax.set_title("Top Features Influencing Prediction", fontsize=16)
            ax.set_xlabel("SHAP Value")
            ax.invert_yaxis()
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01 if width > 0 else width - 0.05,
                        bar.get_y() + bar.get_height()/2,
                        f"{width:.3f}",
                        va='center', ha='left' if width > 0 else 'right')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            plt.close()
            bar_chart_img = buf.getvalue()
            
            # Explanation summary text
            highest_risk = shap_df.iloc[0]
            summary_text = (
                f"The most influential factor contributing to this prediction was "
                f"**{highest_risk['Feature']}**, with a value of **{highest_risk['Value']:.2f}**, "
                f"indicating a strong association with {'periodontitis' if highest_risk['SHAP'] > 0 else 'non-periodontitis'} risk."
            )
            
            st.markdown("### 🔍 Explanation Summary")
            st.markdown(summary_text, unsafe_allow_html=True)
            
            # --- Generate Final Summary Report PNG ---
            font_title_size = 72
            font_body_size = 40
            try:
                font_title = ImageFont.truetype("arial.ttf", font_title_size)
                font_body = ImageFont.truetype("arial.ttf", font_body_size)
            except:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
            
            report_width = 1400
            line_height = 90
            padding = 50
            text_lines = 2  # For summary_text
            report_height = padding + (line_height * (4 + text_lines)) + 340
            
            report_img = Image.new("RGB", (report_width, report_height), "white")
            draw = ImageDraw.Draw(report_img)
            draw.text((padding, padding), "Periodontitis Prediction Report", fill="black", font=font_title)
            draw.text((padding, padding + line_height), f"Patient ID: {selected_id}", fill="black", font=font_body)
            draw.text((padding, padding + line_height*2), f"Prediction: {pred_label}", fill=color, font=font_body)
            draw.text((padding, padding + line_height*3), f"Probability: {pred_prob:.4f}", fill="black", font=font_body)
            draw.text((padding, padding + line_height*4), "Explanation:", fill="black", font=font_body)
            
            # Draw wrapped summary text
            from textwrap import wrap
            wrapped_text = wrap(summary_text, width=70)
            for i, line in enumerate(wrapped_text):
                draw.text((padding, padding + line_height * (5 + i)), line, fill="black", font=font_body)
            
            y_offset = padding + line_height * (5 + len(wrapped_text))
            shap_img = Image.open(io.BytesIO(bar_chart_img)).resize((1200, 300))
            report_img.paste(shap_img, (padding, y_offset))
            y_offset += shap_img.size[1] + 40
            
            img_io = io.BytesIO()
            report_img.save(img_io, format='PNG')
            img_io.seek(0)
            
            # QR Code
            qr = qrcode.make("https://your_hosting_url/your_report_placeholder")
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format='PNG')
            qr_buf.seek(0)
            
            st.subheader("\U0001F4DD Full Summary Report")
            st.image(img_io, caption="Complete Prediction Report")
            st.download_button("⬇️ Download Full Report as PNG", data=img_io, file_name=f"Periodontitis_Report_{selected_id}.png", mime="image/png")
            st.image(qr_buf, caption="\U0001F4F2 Scan QR to Access Report")





















