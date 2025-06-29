# Dunklytics 🏀  
*A Deep Learning-Powered Basketball Prediction Engine*

**Dunklytics** is an AI-driven project that predicts basketball game outcomes using a sophisticated hybrid model. The entire pipeline, from data processing to final prediction, is wrapped in a live, interactive web application deployed on Streamlit Cloud.

---

## 🚀 Features

- **Hybrid Prediction Model:** Utilizes a combination of an **LSTM** network to forecast future team performance based on recent trends and a **Random Forest** model to predict the final game outcome.
- **Tiered Fallback System:** The application intelligently handles missing data. If head-to-head history is unavailable, it gracefully falls back to using overall season averages, ensuring a prediction is almost always possible.
- **Data-Driven Team Rankings:** Employs an **XGBoost** model to generate intelligent team rankings that go beyond simple win-loss records.
- **Synthetic Data Augmentation:** The project includes a robust pipeline for generating realistic synthetic game data using Monte Carlo simulations and other perturbation techniques.
- **Interactive Web Interface:** A user-friendly front-end built with **Streamlit** that allows anyone to select a matchup and get a live prediction.

---

## 🧪 Tech Stack

- **Python:** 3.12  
- **Core Libraries:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Deep Learning:** TensorFlow (Keras)  
- **Web App:** Streamlit  
- **Model Persistence:** Joblib  

---

## 📂 Final Project Structure

The project is structured as a deployable Streamlit application. The key files are located in the root directory for easy deployment.
```
Dunklytics/
├── 📄 app.py                                    # The main Streamlit application script
├── 📄 requirements.txt                          # The list of Python dependencies
├── 🤖 random_forest_model.pkl                   # Trained Random Forest model
├── 🤖 lstm_stat_predictor.h5                    # Trained LSTM model
├── 🧑‍🔧 scaler.pkl                                # The saved data scaler for the LSTM
├── 🧑‍🔧 team_encoder.pkl                          # The saved team name encoder
└── 🏀 basketball_matches_with_opponents.csv     # The primary data file for the app
```
---

## 📌 How to Run the Application

You can run the web application on your local machine:

```bash
# Clone the repo
git clone https://github.com/khatrisahil1/Dunklytics.git
cd Dunklytics

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
# For macOS users, this special command may be needed to fix the XGBoost error locally:
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib streamlit run app.py

# For other systems:
streamlit run app.py
```

## 📈 Sample Result

The application provides a clean interface for predicting game outcomes, complete with tiered logic notifications.
- Model Accuracy: The Random Forest model achieved 100% accuracy on the test set during re-training.
  
`(Note: This is likely due to data leakage in the original data structure and would require more rigorous cross-validation for a production-ready assessment.)`

___
## 📚 Future Goals
The project has a strong foundation with several exciting paths for future development:
-	🏆 Integrate Live Data: Connect the application to a live sports API (e.g., BallDon’tLie, NBA Stats API) to fetch real-time data and keep predictions current.
-	🧠 Model Explainability (XAI): Implement libraries like SHAP or LIME to explain why the model made a certain prediction, showing which stats were most influential.
-	🔁 Automated Re-training Pipeline: Create a scheduled workflow (e.g., using GitHub Actions) to automatically fetch new data and re-train the models weekly to prevent model drift.
-	📊 Enhanced UI/UX: Expand the app into a multi-page dashboard with pages for deep-dive team analysis, player stats, and richer data visualizations.

⸻

##  🙌 Contributions

Feel free to open issues, suggest features, or submit PRs — contributions are welcome!

⸻

## 📄 License

MIT License © SAHIL KHATRI (khatrisahil1)

---
