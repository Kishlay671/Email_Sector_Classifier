import gradio as gr
import pandas as pd
import joblib
import re
import os
from io import StringIO
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the trained model and components with error handling
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ Model files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    raise e

# Text cleaning function (same as training)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove links
    text = re.sub(r"<.*?>", " ", text)                # remove html tags
    text = re.sub(r"[^a-z\s]", " ", text)             # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function for single email
def predict_single_email(email_text):
    if not email_text.strip():
        return "Please enter an email text."
    
    try:
        # Clean the email text
        cleaned_email = clean_text(email_text)
        
        # Vectorize the email
        email_vector = vectorizer.transform([cleaned_email])
        
        # Make prediction
        prediction = model.predict(email_vector)[0]
        predicted_sector = label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities (if supported by the model)
        try:
            probabilities = model.decision_function(email_vector)[0]
            # Get top 3 predictions
            top_indices = probabilities.argsort()[-3:][::-1]
            top_sectors = label_encoder.inverse_transform(top_indices)
            top_scores = probabilities[top_indices]
            
            result = f"**Predicted Sector: {predicted_sector}**\n\n"
            result += "**Top 3 Predictions:**\n"
            for i, (sector, score) in enumerate(zip(top_sectors, top_scores)):
                result += f"{i+1}. {sector}: {score:.3f}\n"
                
        except:
            result = f"**Predicted Sector: {predicted_sector}**"
            
        return result
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Prediction function for CSV file
def predict_csv_file(file):
    if file is None:
        return None, "Please upload a CSV file."
    
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.name)
        
        # Check if 'Emails' column exists
        if 'Emails' not in df.columns:
            return None, "Error: CSV file must contain an 'Emails' column."
        
        # Clean the email texts
        df['Cleaned_Emails'] = df['Emails'].apply(clean_text)
        
        # Make predictions
        email_vectors = vectorizer.transform(df['Cleaned_Emails'])
        predictions = model.predict(email_vectors)
        predicted_sectors = label_encoder.inverse_transform(predictions)
        
        # Add predictions to dataframe
        df['Predicted_Sector'] = predicted_sectors
        
        # Create output CSV
        output_csv = StringIO()
        df[['Emails', 'Predicted_Sector']].to_csv(output_csv, index=False)
        output_csv.seek(0)
        
        # Save to temporary file for download
        output_filename = "predictions.csv"
        df[['Emails', 'Predicted_Sector']].to_csv(output_filename, index=False)
        
        success_msg = f"✅ Successfully processed {len(df)} emails. Download the results below."
        
        return output_filename, success_msg
        
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"

# Get available sectors for display
available_sectors = list(label_encoder.classes_)
sectors_text = ", ".join(available_sectors)

# Create Gradio interface
with gr.Blocks(title="Email Sector Classification", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 📧 Email Sector Classification")
    gr.Markdown("Classify emails into business sectors using machine learning.")
    
    gr.Markdown(f"**Available Sectors:** {sectors_text}")
    
    with gr.Tabs():
        # Tab 1: Single Email Prediction
        with gr.Tab("Single Email Prediction"):
            gr.Markdown("### Enter an email to classify its sector")
            
            with gr.Row():
                with gr.Column(scale=2):
                    email_input = gr.Textbox(
                        label="Email Text",
                        placeholder="Enter your email content here...",
                        lines=8,
                        max_lines=15
                    )
                    predict_btn = gr.Button("Predict Sector", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    prediction_output = gr.Markdown(label="Prediction Result")
            
            predict_btn.click(
                fn=predict_single_email,
                inputs=email_input,
                outputs=prediction_output
            )
            
            # Example emails
            gr.Markdown("### Example Emails")
            examples = [
                "We are looking for experienced software developers to join our tech team. Requirements include Python, JavaScript, and cloud technologies.",
                "Our hospital is seeking qualified nurses for the emergency department. Must have current RN license and BLS certification.",
                "Join our sales team! We offer competitive commission rates and comprehensive training for motivated individuals.",
                "We provide comprehensive financial planning services including investment management and retirement planning."
            ]
            
            gr.Examples(
                examples=examples,
                inputs=email_input,
                outputs=prediction_output,
                fn=predict_single_email,
                cache_examples=True
            )
        
        # Tab 2: CSV File Upload
        with gr.Tab("Batch CSV Processing"):
            gr.Markdown("### Upload a CSV file with emails to classify")
            gr.Markdown("**CSV Format:** Your file should have an 'Emails' column containing the email texts.")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"],
                        file_count="single"
                    )
                    process_btn = gr.Button("Process CSV", variant="primary", size="lg")
                
                with gr.Column():
                    file_status = gr.Markdown()
                    download_file = gr.File(label="Download Results", visible=False)
            
            def process_and_update(file):
                result_file, status = predict_csv_file(file)
                if result_file:
                    return status, gr.File(value=result_file, visible=True)
                else:
                    return status, gr.File(visible=False)
            
            process_btn.click(
                fn=process_and_update,
                inputs=file_input,
                outputs=[file_status, download_file]
            )
            
            # CSV format example
            gr.Markdown("### CSV Format Example")
            gr.Markdown("""
            ```
            Emails
            "We are hiring software engineers with Python experience"
            "Our clinic needs registered nurses for patient care"
            "Looking for sales representatives in the automotive industry"
            ```
            """)

    gr.Markdown("---")
    gr.Markdown("*Powered by scikit-learn and Gradio*")

# Launch the app
if __name__ == "__main__":
    demo.launch()