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

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("XAI_binary_model_ori.h5")

model = load_model()

# Function to generate SHAP summary plots for the entire dataset
def generate_dataset_shap_summary(df, X_scaled, explainer):
    st.subheader("📊 Dataset-wide SHAP Analysis")
    
    # Generate SHAP values for the entire dataset
    shap_values = explainer(X_scaled)
    
    # Create tabs for different types of visualization
    tab1, tab2 = st.tabs(["Overall Summary", "Domain-Specific Effects"])
    
    with tab1:
        st.markdown("### Overall Feature Importance")
        
        # Save the overall summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values.values, X_scaled, feature_names=[feature_name_map.get(name, name) for name in df.columns], show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        st.image(buf)
        
        # Beeswarm plot with absolute values
        st.markdown("### Feature Impact Magnitude")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values.values, X_scaled, feature_names=[feature_name_map.get(name, name) for name in df.columns], plot_type="bar", show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        st.image(buf)
    
    with tab2:
        # Create domain-specific SHAP summary plots
        for group_name, group_features in feature_groups.items():
            st.markdown(f"### {group_name} Features")
            
            # Get indices of features in this group
            group_indices = [i for i, name in enumerate(df.columns) if name in group_features]
            
            if not group_indices:
                st.write("No features found in this group.")
                continue
            
            # Extract feature names for this group
            group_feature_names = [feature_name_map.get(df.columns[i], df.columns[i]) for i in group_indices]
            
            # Create SHAP summary plot for this group
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(
                shap_values.values[:, group_indices], 
                X_scaled[:, group_indices],
                feature_names=group_feature_names,
                show=False
            )
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            st.image(buf)
            
            # Add a bar plot showing feature importance within this domain
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(
                shap_values.values[:, group_indices], 
                X_scaled[:, group_indices],
                feature_names=group_feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            st.image(buf)
            
            # Add feature interaction analysis for the top feature in this group
            if len(group_indices) > 1:
                # Find the most important feature in this group
                mean_abs_shap = np.mean(np.abs(shap_values.values[:, group_indices]), axis=0)
                top_feature_idx = group_indices[np.argmax(mean_abs_shap)]
                top_feature_name = feature_name_map.get(df.columns[top_feature_idx], df.columns[top_feature_idx])
                
                st.markdown(f"#### Feature Interactions with {top_feature_name}")
                
                # Create dependence plot for the top feature in this group
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.dependence_plot(
                    top_feature_idx, 
                    shap_values.values, 
                    X_scaled,
                    feature_names=[feature_name_map.get(name, name) for name in df.columns],
                    show=False
                )
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                st.image(buf)

# Main interface for file upload
uploaded_file = st.file_uploader("Upload Excel file with patient data", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    if "ID" not in df.columns:
        st.error("The uploaded file must contain an 'ID' column.")
    else:
        # Add tabs for individual patient analysis and dataset-wide analysis
        tab1, tab2 = st.tabs(["Individual Patient Analysis", "Dataset-wide Analysis"])
        
        with tab1:
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
                    Line2D([0], [0], color='green', lw=4, label='Protective Factor'),
                    Line2D([0], [0], color='red', lw=4, label='Risk Factor')
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
        
        with tab2:
            # Process all data for dataset-wide analysis
            X_raw = df[raw_features]
            X_encoded = pd.get_dummies(X_raw)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            
            # Generate SHAP background
            background = shap.sample(X_scaled, 100, random_state=42)
            explainer = shap.Explainer(model, background)
            
            # Call function to create dataset-wide SHAP summary plots
            generate_dataset_shap_summary(X_encoded, X_scaled, explainer)
            
            # Add dataset statistics
            st.subheader("📊 Dataset Statistics")
            
            # Calculate predictions for all patients
            all_probs = model.predict(X_scaled).flatten()
            all_labels = ["Periodontitis" if p >= 0.5 else "Non-Periodontitis" for p in all_probs]
            
            # Display distribution of predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prediction Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(all_probs, bins=20, color='skyblue', edgecolor='black')
                ax.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
                ax.set_xlabel('Probability of Periodontitis')
                ax.set_ylabel('Number of Patients')
                ax.legend()
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                plt.close()
                st.image(buf)
            
            with col2:
                st.markdown("### Prediction Counts")
                label_counts = pd.Series(all_labels).value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
                ax.axis('equal')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                plt.close()
                st.image(buf)
            
            # Feature statistics
            st.markdown("### Feature Statistics")
            
            # Display feature correlations
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = X_raw.corr()
            im = ax.imshow(corr, cmap='coolwarm')
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels([feature_name_map.get(col, col) for col in corr.columns], rotation=45, ha='right')
            ax.set_yticklabels([feature_name_map.get(col, col) for col in corr.columns])
            plt.colorbar(im)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            plt.close()
            st.image(buf, caption="Feature Correlation Matrix")
            
            # Display downloadable insights
            st.subheader("📥 Download Dataset Analysis")
            
            # Create a summary PDF report of dataset analysis
            from matplotlib.backends.backend_pdf import PdfPages
            
            def create_dataset_report():
                buf = io.BytesIO()
                with PdfPages(buf) as pdf:
                    # Title page
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.5, "Periodontitis Dataset Analysis", 
                           fontsize=24, ha='center', va='center')
                    ax.text(0.5, 0.4, f"Total Patients: {len(df)}", 
                           fontsize=18, ha='center', va='center')
                    ax.text(0.5, 0.3, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}", 
                           fontsize=18, ha='center', va='center')
                    ax.axis('off')
                    pdf.savefig(fig)
                    plt.close()
                    
                    # Overall SHAP summary plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap_values = explainer(X_scaled)
                    shap.summary_plot(shap_values.values, X_scaled, 
                                     feature_names=[feature_name_map.get(name, name) for name in X_encoded.columns], 
                                     show=False)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    
                    # Domain-specific plots
                    for group_name, group_features in feature_groups.items():
                        group_indices = [i for i, name in enumerate(X_encoded.columns) if name in group_features]
                        if not group_indices:
                            continue
                            
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.set_title(f"{group_name} Features Impact", fontsize=16)
                        shap.summary_plot(
                            shap_values.values[:, group_indices], 
                            X_scaled[:, group_indices],
                            feature_names=[feature_name_map.get(X_encoded.columns[i], X_encoded.columns[i]) for i in group_indices],
                            show=False
                        )
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()
                
                buf.seek(0)
                return buf
            
            dataset_report = create_dataset_report()
            st.download_button(
                "⬇️ Download Dataset Analysis Report",
                data=dataset_report,
                file_name="Periodontitis_Dataset_Analysis.pdf",
                mime="application/pdf"
            )































