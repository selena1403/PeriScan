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
        
        # Save the overall summary plot for the dataset
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
        # Create domain-specific SHAP summary plots for each group of features
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
            # Handle individual patient analysis (same as before)
            pass
        
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
            generate_dataset_shap_summary(df, X_scaled, explainer)
            
            # Add dataset statistics (same as before)
            pass


































