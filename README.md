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

The dataset comes from the **University of Wisconsin** (1995) and contains
569 real cases: 357 benign and 212 malignant. A cleaned mirror of this
dataset is available on Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
## Disclaimer

This is a prototype for educational purposes only. It is not a medical
diagnostic tool.

---

## License

This project (source code and documentation) is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). See https://creativecommons.org/licenses/by-nc-sa/4.0/ for
the full license text.

Source code files include a short header with the same license. Third-party
dependencies listed in `requirements.txt` retain their own licenses and are
not automatically re-licensed by this project; review them individually if
you plan to redistribute derivatives.
