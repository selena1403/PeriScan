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
from matplotlib.lines import Line2D  # For the custom legend

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

feature_name_map = {
    'sex': 'Sex', 'age': 'Age', 'bmi': 'BMI', 'pulse': 'Pulse', 'sbpL': 'Systolic BP', 'dbpL': 'Diastolic BP',
    'wbc': 'White Blood Cells', 'rbc': 'Red Blood Cells', 'hb': 'Hemoglobin', 'hct': 'Hematocrit', 'plt': 'Platelets',
    't_chol': 'Total Cholesterol', 'hdl': 'HDL', 'ldl': 'LDL',
    'dental_1': 'Chewing Discomfort Score', 'teeth_3': 'Remaining Teeth Count', 'teeth_problem': 'Problematic Teeth Count'
}

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

            st.subheader("🔍 SHAP Force Plots by Feature Group")
            shap.initjs()
            background = shap.sample(X_scaled, 100, random_state=42)
            explainer = shap.Explainer(model, background)
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

            # --- SHAP Bar Plot with Legend ---
            fig, ax = plt.subplots(figsize=(10, 6))

            # Bar plot
            bars = ax.barh(shap_df["Feature"], shap_df["SHAP"], color=shap_df["Color"])

            # Title and labels
            ax.set_title("Top Features Influencing Prediction", fontsize=16)
            ax.set_xlabel("SHAP Value")
            ax.invert_yaxis()

            # Add legend explicitly
            legend_elements = [
                Line2D([0], [0], color='green', lw=4, label='Reduces Risk'),
                Line2D([0], [0], color='red', lw=4, label='Increases Risk')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

            # Adjust layout to prevent clipping
            plt.tight_layout()

            # Save plot to buffer
            bar_buf = io.BytesIO()
            plt.savefig(bar_buf, format='png', dpi=300)
            bar_buf.seek(0)
            bar_img = bar_buf.getvalue()
            plt.close()

            image_buffers = []
            counter = 0
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
                image_buffers.append((f"{group_name} SHAP", buf.getvalue()))

                if counter == 1:
                    image_buffers.append(("SHAP Bar Chart", bar_img))
                counter += 1

            st.markdown("### 🔍 Explanation Summary")
            summary_lines = [f"Rank {i+1}: **{row['Feature']}** ({row['Value']:.2f}) - {'↑ Increases' if row['SHAP'] > 0 else '↓ Decreases'} risk"
                             for i, (_, row) in enumerate(shap_df.iterrows())]
            high_risk = shap_df.iloc[0]
            warning_text = f"⚠️ Most influential risk factor: **{high_risk['Feature']}**. Monitor and manage closely."
            summary_text = "\n".join(summary_lines) + f"\n\n{warning_text}"
            st.markdown(summary_text, unsafe_allow_html=True)

            # --- Larger Text Final Summary Report ---
            font_title_size = 42
            font_body_size = 32

            try:
                font_title = ImageFont.truetype("arial.ttf", font_title_size)
                font_body = ImageFont.truetype("arial.ttf", font_body_size)
            except:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_title_size)
                font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_body_size)

            line_height = 90
            padding = 80
            text_lines = 5
            shap_image_height = 1000
            report_width = 1800
            report_height = padding * 2 + line_height * text_lines + shap_image_height + 60

            report_img = Image.new("RGB", (report_width, report_height), "white")
            draw = ImageDraw.Draw(report_img)

            draw.text((padding, padding), "🦷 Periodontitis Prediction Report", fill="black", font=font_title)
            draw.text((padding, padding + line_height * 1), f"Patient ID: {selected_id}", fill="black", font=font_body)
            draw.text((padding, padding + line_height * 2), f"Prediction: {pred_label}", fill=color, font=font_body)
            draw.text((padding, padding + line_height * 3), f"Probability: {pred_prob:.4f}", fill="black", font=font_body)

            summary_text = (
                f"Top contributing factor: {high_risk['Feature']} "
                f"(value: {high_risk['Value']:.2f}) had the highest impact on the prediction."
            )
            draw.text((padding, padding + line_height * 4), summary_text, fill="black", font=font_body)

            y_offset = padding + line_height * text_lines
            shap_img = Image.open(io.BytesIO(bar_buf.getvalue())).resize((report_width - 2 * padding, shap_image_height))
            report_img.paste(shap_img, (padding, y_offset))

            img_io = io.BytesIO()
            report_img.save(img_io, format='JPEG')
            img_io.seek(0)

            qr = qrcode.make("https://your_hosting_url/your_report_placeholder")
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format='PNG')
            qr_buf.seek(0)

            st.subheader("📝 Summary Report")
            st.image(img_io, caption="Complete Prediction Report")
            st.download_button("⬇️ Download Report", data=img_io, file_name=f"Periodontitis_Report_{selected_id}.jpg", mime="image/jpeg")
            st.image(qr_buf, caption="📱 Scan QR to Access Report")

























