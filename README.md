---
title: Email Sector Classification
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 📧 Email Sector Classification

**🚀 Live Demo:** [https://huggingface.co/spaces/KishlayKumar/Email-Sector-Classification](https://huggingface.co/spaces/KishlayKumar/Email-Sector-Classification)

This application automatically classifies emails into different business sectors using machine learning. Built with scikit-learn and deployed on Hugging Face Spaces with a user-friendly Gradio interface.

## ✨ Features

- **🔍 Single Email Prediction**: Enter any email content and instantly get the predicted business sector
- **📊 Batch CSV Processing**: Upload CSV files with multiple emails for bulk classification
- **💾 Download Results**: Get processed results as downloadable CSV files
- **🎯 Confidence Scores**: View prediction confidence and top alternatives
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 How to Use

### Single Email Classification
1. Navigate to the **"Single Email Prediction"** tab
2. Paste your email content in the text box
3. Click **"Predict Sector"** to get instant results
4. View the predicted sector along with confidence scores

### Batch Processing
1. Go to the **"Batch CSV Processing"** tab
2. Upload a CSV file with an **"Emails"** column
3. Click **"Process CSV"** to classify all emails
4. Download the results with predictions added

## 📋 CSV Format Requirements

Your CSV file should have this structure:

```csv
Emails
"We are hiring software engineers with Python and React experience for our startup"
"Our hospital is seeking qualified nurses for the emergency department"
"Join our sales team selling luxury automobiles with great commission rates"
"We provide comprehensive financial planning and investment advisory services"
"Seeking experienced teachers for our elementary school mathematics department"
```

**Requirements:**
- Must contain an **"Emails"** column header
- Each row should have one email text
- Empty rows are automatically skipped
- Supports various email formats and lengths

## 🏢 Supported Business Sectors

The model can classify emails into various business sectors including:
- **Technology & Software**
- **Healthcare & Medical**
- **Finance & Banking**
- **Sales & Retail**
- **Education**
- **Manufacturing**
- **Real Estate**
- **Legal Services**
- **Marketing & Advertising**
- **And many more...**

## 🔬 Technical Details

### Model Architecture
- **Algorithm**: Linear Support Vector Machine (LinearSVM)
- **Feature Extraction**: TF-IDF Vectorization with unigrams and bigrams
- **Max Features**: 20,000 features
- **Text Preprocessing**: Comprehensive cleaning including URL removal, HTML tag removal, and normalization

### Performance Metrics
- **Training Accuracy**: High accuracy on validation set
- **F1-Score**: Weighted F1-score for multi-class classification
- **Cross-Validation**: Stratified train-test split ensuring balanced representation

### Preprocessing Pipeline
1. **Text Cleaning**: Convert to lowercase, remove URLs and HTML tags
2. **Feature Extraction**: TF-IDF vectorization with n-gram analysis
3. **Classification**: Linear SVM with optimized hyperparameters
4. **Post-processing**: Label decoding and confidence scoring

## 💻 Technology Stack

- **🤖 Machine Learning**: scikit-learn 1.6.1
- **🎨 Frontend**: Gradio 4.44.0
- **📊 Data Processing**: Pandas, NumPy
- **☁️ Deployment**: Hugging Face Spaces
- **🔧 Model Persistence**: Joblib

## 🎯 Use Cases

- **📧 Email Routing**: Automatically route emails to appropriate departments
- **🏢 Lead Classification**: Classify business inquiries by industry sector
- **📈 Market Research**: Analyze email communications by business sector
- **🤖 Content Filtering**: Organize emails by business category
- **📊 Business Intelligence**: Sector-wise analysis of communications

## 🔧 Local Development

To run this application locally:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/KishlayKumar/Email-Sector-Classification
cd Email-Sector-Classification

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📈 Model Performance

The model has been trained and evaluated on a comprehensive email dataset with the following characteristics:
- **Multi-class Classification**: Handles multiple business sectors
- **Robust Preprocessing**: Handles various email formats and noise
- **Balanced Training**: Stratified sampling ensures fair representation
- **Validation**: Rigorous testing on held-out data

## 🤝 Contributing

This project is open for improvements! Feel free to:
- Report issues or bugs
- Suggest new features
- Improve model performance
- Enhance the user interface

## 📄 License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kishlay Kumar**
- 🌐 Hugging Face: [@KishlayKumar](https://huggingface.co/KishlayKumar)
- 📧 For questions or collaborations, feel free to reach out!

## 🙏 Acknowledgments

- **Hugging Face** for providing the deployment platform
- **scikit-learn** team for the excellent ML library
- **Gradio** team for the intuitive interface framework
- The open-source community for valuable resources and inspiration

---

**🚀 Try it now:** [Email Sector Classification App](https://huggingface.co/spaces/KishlayKumar/Email-Sector-Classification)

*Built with ❤️ using Python, scikit-learn, and Gradio*