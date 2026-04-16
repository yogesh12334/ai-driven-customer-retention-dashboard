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
git clone (https://github.com/yogesh12334/ai-driven-customer-retention-dashboard)
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


_________________________________________________________________________________________________________________________

<img width="1903" height="669" alt="image" src="https://github.com/user-attachments/assets/801f416b-f0f0-4b05-890a-868f37568bd1" />

_________________________________________________________________________________________________________________________

<img width="1898" height="848" alt="image" src="https://github.com/user-attachments/assets/653b06c9-6a69-4c6f-b0b6-5a08b0696275" />

_________________________________________________________________________________________________________________________

<img width="1893" height="542" alt="image" src="https://github.com/user-attachments/assets/7e740ade-29c6-4314-aa8c-7333df2e54ef" />

_________________________________________________________________________________________________________________________

<img width="1784" height="526" alt="image" src="https://github.com/user-attachments/assets/cb176645-e0d0-462e-a18d-d0196f08907f" />

_________________________________________________________________________________________________________________________

<img width="1775" height="335" alt="image" src="https://github.com/user-attachments/assets/554c35df-d19b-479a-83b5-0981f2177b68" />

________________________________________________________________________________________________________________________

<img width="1206" height="858" alt="image" src="https://github.com/user-attachments/assets/689d4157-2bff-49ed-b075-97cb9a39752d" />
