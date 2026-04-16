🚀 Churn Intelligence System v3.0
Churn Intelligence System v3.0 is an industrial-grade analytics platform designed for subscription-based businesses (Telecom/SaaS) to identify and prevent customer churn. Unlike standard models, this system integrates SHAP (Explainable AI) to provide transparency into why a customer is likely to leave, turning predictions into actionable business insights.
____________________________________________________________________________________________________________________________

📽️ Dashboard Preview

http://localhost:8501/

___________________________________________________________________________________________________________________________

🛠️ Key Features
Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to ensure model transparency and build stakeholder trust.
Real-time Prediction Engine: Powered by a FastAPI backend that serves the machine learning model for live batch scoring.
Business ROI Calculator: A dedicated module that translates ML probabilities into financial impact, estimating potential Revenue Recovery (₹).
Modular Architecture: The entire system is configuration-driven via config.yaml. Zero hard-coding policy for maximum scalability.
Industrial UI/UX: A high-performance Streamlit dashboard featuring custom CSS for a professional dark-mode industrial aesthetic.
_________________________________________________________________________________________________________________________

🏗️ System Architecture
The project is divided into three core layers:
Backend (The Engine): api.py (FastAPI + Uvicorn) - Handles RESTful requests and live model inference.
Frontend (The Interface): dashboard.py (Streamlit) - Multi-tabbed reporting, deep-dive analytics, and live scoring.
Config (The Brain): config.yaml - Centralized management of model paths, database settings, and business thresholds.

_________________________________________________________________________________________________________________________

💻 Tech Stack

Category	              Tools
Languages	              Python 3.13
Machine Learning	      XGBoost, Random Forest, Scikit-learn, SHAP
Data Handling	          Pandas, Numpy, SQLite3
Web Frameworks	        FastAPI, Streamlit
Format & Styling	      YAML, Custom CSS, Plotly

_________________________________________________________________________________________________________________________

🚀 Installation & Setup

1 Clone the Repository:
git clone https://github.com/your-username/churn-intelligence-v3.git
cd churn-intelligence-v3

2 Install Dependencies:
pip install -r requirements.txt

3 Start the Backend API:
python api.py

4 Launch the Dashboard:
streamlit run dashboard.py

_________________________________________________________________________________________________________________________

📂 Project Structure

├── api.py              # FastAPI Backend Inference Engine
├── dashboard.py        # Streamlit Analytics Interface
├── config.yaml         # Centralized System Configurations
├── train_model.py      # ML Model Training Pipeline
├── database.db         # SQLite Storage for Customer Records
├── models/             # Serialized Models & SHAP Metadata
├── predictions/        # Generated Insights and Batch Results
└── requirements.txt    # Project Dependencies & Libraries

_________________________________________________________________________________________________________________________

📈 Impact Analysis

The model was tested on a dataset of 300,000+ customers, achieving a robust ~77% AUC-ROC. According to the integrated ROI Calculator, a mere 5% reduction in churn using this tool has the potential to recover approximately ₹88.2 Crores in revenue.

_________________________________________________________________________________________________________________________

👤 Author
Yogesh Kumar
Aspiring Data & Business Analyst
LinkedIn Profile (https://www.linkedin.com/in/yogesh-kumar-saini/)


