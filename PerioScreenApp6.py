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

# --- Streamlit UI ---
st.set_page_config(page_title="Periodontitis Risk Report", layout="wide")
st.title("\U0001F9B7 Periodontitis Prediction and SHAP Explanation")
st.markdown("<h2 style='text-align: center;'>Periodontitis Risk Prediction and Detailed SHAP Explanation</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel file with patient data", type=["xlsx"])
model = tf.keras.models.load_model("XAI_binary_model_ori.h5")

# --- Feature definitions ---
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
feature_name_map = {
    'sex': 'Sex', 'age': 'Age', 'bmi': 'BMI', 'pulse': 'Pulse', 'sbpL': 'Systolic BP', 'dbpL': 'Diastolic BP',
    'wbc': 'White Blood Cells', 'rbc': 'Red Blood Cells', 'hb': 'Hemoglobin', 'hct': 'Hematocrit', 'plt': 'Platelets',
    't_chol': 'Total Cholesterol', 'hdl': 'HDL', 'ldl': 'LDL',
    'dental_1': 'Chewing Discomfort Score', 'teeth_3': 'Remaining Teeth Count', 'teeth_problem': 'Problematic Teeth Count'
}

# --- Main ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "ID" not in df.columns:
        st.error("The uploaded file must contain an 'ID' column.")
    else:
        selected_id = st.selectbox("Select Patient ID", df["ID"])
        selected_patient = df[df["ID"] == selected_id]
        X_raw = df[raw_features]
        X_encoded = pd.get_dummies(X_raw)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        input_raw = selected_patient[raw_features]
        input_encoded = pd.get_dummies(input_raw).reindex(columns=X_encoded.columns, fill_value=0)
        input_scaled = scaler.transform(input_encoded)

        pred_prob = model.predict(input_scaled)[0][0]
        pred_label = "Periodontitis" if pred_prob >= 0.5 else "Non-Periodontitis"
        color = "red" if pred_prob >= 0.5 else "green"

        st.markdown(f"### Prediction: **<span style='color:{color};'>{pred_label}</span>**", unsafe_allow_html=True)
        st.markdown(f"#### Probability: **<span style='color:{color};'>{pred_prob:.4f}</span>**", unsafe_allow_html=True)

        shap.initjs()
        explainer = shap.Explainer(model, shap.sample(X_scaled, 100, random_state=42))
        shap_values = explainer(input_scaled)

        shap_vals = shap_values.values[0]
        input_vals = input_encoded.iloc[0].values
        base_value = shap_values.base_values[0]
        feature_names = input_encoded.columns.tolist()
        friendly_feature_names = [feature_name_map.get(name, name) for name in feature_names]

        shap_df = pd.DataFrame({
            "Feature": friendly_feature_names,
            "Value": input_vals,
            "SHAP": shap_vals
        })
        shap_df["Abs_SHAP"] = shap_df["SHAP"].abs()
        shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)
        shap_df["Color"] = shap_df["SHAP"].apply(lambda x: "red" if x > 0 else "green")

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=shap_df["Color"])
        ax.set_title("Top Features Influencing Prediction", fontsize=18)
        ax.set_xlabel("SHAP Value", fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.invert_yaxis()
        plt.tight_layout()

        bar_buf = io.BytesIO()
        plt.savefig(bar_buf, format='png', dpi=300, bbox_inches='tight')
        bar_buf.seek(0)
        bar_img = bar_buf.getvalue()
        bar_chart = Image.open(io.BytesIO(bar_img))
        plt.close()

        st.image(bar_buf, caption="Top SHAP Features")

        # --- Report Generation ---
        try:
            font_title = ImageFont.truetype("arial.ttf", 900)
            font_body = ImageFont.truetype("arial.ttf", 700)
        except:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()

        report_width = 1600
        padding = 60
        line_height = 100
        report_height = padding + 5 * line_height + 360
        report_img = Image.new("RGB", (report_width, report_height), "white")
        draw = ImageDraw.Draw(report_img)

        draw.text((padding, padding), "Periodontitis Prediction Report", font=font_title, fill="black")
        draw.text((padding, padding + line_height), f"Patient ID: {selected_id}", font=font_body, fill="black")
        draw.text((padding, padding + 2 * line_height), f"Prediction: {pred_label}", font=font_body, fill=color)
        draw.text((padding, padding + 3 * line_height), f"Probability: {pred_prob:.4f}", font=font_body, fill="black")

        top_feature = shap_df.iloc[0]
        summary_line = f"Top contributing factor: {top_feature['Feature']} (value: {top_feature['Value']:.2f})"
        draw.text((padding, padding + 4 * line_height), summary_line, font=font_body, fill="black")

        # Add bar chart to report
        shap_img = Image.open(io.BytesIO(bar_img)).resize((1200, 300))
        report_img.paste(shap_img, (padding, padding + 5 * line_height))

        img_io = io.BytesIO()
        report_img.save(img_io, format='PNG')
        img_io.seek(0)

        st.subheader("📝 Final Summary Report")
        st.image(img_io, caption="Complete Prediction Report")
        st.download_button("⬇️ Download Full Report as PNG", data=img_io,
                           file_name=f"Periodontitis_Report_{selected_id}.png", mime="image/png")

        # QR code
        qr = qrcode.make("https://your_hosting_url/your_report_placeholder")
        qr_buf = io.BytesIO()
        qr.save(qr_buf, format='PNG')
        qr_buf.seek(0)
        st.image(qr_buf, caption="📱 Scan QR to Access Report")






















