# Breast Cancer Diagnosis Predictor

A prototype web application that predicts whether a breast tumor is
**benign** or **malignant** based on cell nucleus measurements, using a
Random Forest classifier trained on the Wisconsin Breast Cancer Dataset.

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

- On first load, the app trains a Random Forest classifier on scikit-learn's
  built-in breast cancer dataset (569 samples, 30 features).
- The model is cached so training only happens once per server session.
- Adjust the 30 input features in the sidebar to get a real-time prediction.
- The main area shows the diagnosis (Benign/Malignant) with a confidence score.
- A radar chart visualizes your input values compared to the dataset average.

## Project Structure

| File                      | Purpose                                    |
|---------------------------|--------------------------------------------|
| `app.py`                  | Streamlit web application                  |
| `model.py`                | Model training, evaluation, and prediction |
| `requirements.txt`        | Python package dependencies                |
| `.streamlit/config.toml`  | Streamlit theme and server configuration   |
| `README.md`               | This file                                  |

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn (Random Forest + StandardScaler pipeline)
- pandas, numpy, matplotlib

## Deployment (Streamlit Cloud)

This app is ready to deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and main file `app.py`.
4. Click **Deploy** -- Streamlit Cloud will install dependencies from
   `requirements.txt` and start the app automatically.

No Dockerfile, Procfile, or server configuration needed.

## Disclaimer

This is a prototype for educational purposes only. It is not a medical
diagnostic tool.
