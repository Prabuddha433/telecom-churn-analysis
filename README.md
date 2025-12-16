# 📊 Telecom Customer Churn Prediction

## 📌 Problem Statement

Customer churn is a major challenge in the telecom industry. Acquiring new customers is significantly more expensive than retaining existing ones. The objective of this project is to **predict whether a telecom customer is likely to churn** based on their usage patterns, services subscribed, and account information. Early churn prediction helps businesses take proactive retention actions.

---

## 💡 Solution Overview

This project builds an **end-to-end machine learning pipeline** to analyze telecom customer data and predict churn. It includes:

- Data cleaning and exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model persistence
- An interactive **Streamlit web application** for real-time churn prediction

---

## 🗂️ Dataset Information

The dataset used in this project is the **Telco Customer Churn Dataset**, which is publicly available on Kaggle.

- **Download link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### How to Use the Dataset
1. Download the dataset (`Telco-Customer-Churn.csv`) from Kaggle.
2. Rename the file to `telecom_churn.csv`.
3. Place the file inside the `data/` directory of this project:
   ```
   data/telecom_churn.csv
   ```
4. The training script and Streamlit app will automatically load this file during execution.

---

## 🗂️ Project File Structure

```
telecom-churn-prediction/
│
├── data/
│   └── telecom_churn.csv        # Raw dataset
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   ├── eda.py                   # Exploratory data analysis
│   ├── model.py                 # Model training & saving logic
│   └── explainability.py        # Feature importance / insights
│
├── model/
│   └── churn_model.pkl          # Trained ML model
│
├── app.py                       # Streamlit application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🧰 Tech Stack & Libraries Used

### 🔹 Tech Stack

- **Programming Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

### 🔹 Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

---

## ⭐ Key Features

- End-to-end ML pipeline
- Clean and modular project structure
- Interactive Streamlit UI
- Automatic model training if model is missing
- Feature importance and explainability
- Easy local execution

---

## ▶️ Step-by-Step: How to Run the Project Locally

### ⚙️ Model Training Information

The project is designed to **automatically train the model if it does not already exist** in the `model/` directory. However, you can also explicitly train the model by running the training script.

- If `model/churn_model.pkl` **exists** → the app loads the saved model
- If `model/churn_model.pkl` **does not exist** → the model is trained automatically using the raw dataset

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd telecom-churn-prediction
```

### 2️⃣ Create & Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (Manual – Optional)

You can manually train and save the model before running the app:

```bash
python -m src.model
```

This will:

- Load the telecom churn dataset
- Perform preprocessing & feature engineering
- Train the machine learning model
- Save the trained model to `model/churn_model.pkl`
- Generate SHAP explainability output.

> 💡 **Note:** This step is optional. If you skip it, the Streamlit app will automatically train the model if it is not found.

### 5️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

```

---

## 👀 What You Will See After Running
- A **web-based dashboard** opened in your browser
- Input fields to enter customer details
- A **churn prediction result** (Yes / No)
- Model-driven insights and feature importance
- Clean and user-friendly UI

---

## 📈 Business Impact
- Helps telecom companies **identify high-risk customers**
- Enables **data-driven retention strategies**
- Reduces revenue loss due to churn
- Improves customer satisfaction and lifetime value

---

## 🧠 Skills Demonstrated
- Data Cleaning & EDA
- Feature Engineering
- Machine Learning Model Building
- Model Evaluation
- Python Project Structuring
- Streamlit App Development
- Git & GitHub Version Control
- Problem Solving & Debugging

---

## 👤 Author
**Prabuddha Ray**  
Bachelor’s in Computer Science Engineering  
Aspiring Data Analyst / Machine Learning Engineer

---

## 📝 Note
This project is built for **learning, demonstration, and portfolio purposes**. The dataset used is publicly available from  and the model performance can be further improved with advanced algorithms and hyperparameter tuning.

---

⭐ *If you find this project helpful, feel free to give it a star!*

```
