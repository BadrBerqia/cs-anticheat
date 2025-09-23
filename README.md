# CS-AntiCheat: A Statistical Approach to Cheat Detection

This project implements a data-driven anti-cheat system for Counter-Strike, using machine learning to identify players with anomalous performance statistics. The system analyzes player data, trains a set of models, and provides a framework for evaluating their performance and business impact.

## Features

*   **Data-Driven Detection:** Uses player statistics to identify potential cheaters.
*   **Ensemble Modeling:** Combines multiple machine learning models (Random Forest, Gradient Boosting) for robust predictions.
*   **Comprehensive Evaluation:** Includes scripts for cross-validation, performance visualization (ROC/PR curves), and business impact analysis.
*   **REST API:** A simple FastAPI endpoint to get a prediction for a player's stats.

## Project Structure

```
.
├── data/                  # Raw and processed data (not in git)
├── final_models/          # Trained and balanced models
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── api/
│   │   └── detection_api.py  # FastAPI detection endpoint
│   ├── models/            # Model training and prediction logic
│   └── utils/             # Database and utility functions
├── comprehensive_evaluation.py # Main script for model evaluation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BadrBerqia/cs-anticheat.git
    cd cs-anticheat
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Comprehensive Evaluation

To run the full evaluation pipeline, which generates performance plots, metrics, and a summary report:

```bash
python comprehensive_evaluation.py
```

The output will be saved in the `evaluation_plots/` directory and `evaluation_report.txt`.

### Running the Detection API

The project includes a simple FastAPI server to expose the detection model as an API endpoint.

1.  **Run the server:**
    ```bash
    uvicorn src.api.detection_api:app --reload
    ```

2.  **Send a request:**
    You can send a POST request to `http://127.0.0.1:8000/predict/` with player data to get a cheat probability score.

    Example using `curl`:
    ```bash
    curl -X "POST" "http://127.0.0.1:8000/predict/" \
         -H 'Content-Type: application/json; charset=utf-8' \
         -d '{
              "avg_kills": 25.5,
              "avg_deaths": 10.1,
              "avg_headshots": 15.2,
              "avg_accuracy": 0.75,
              "max_kills": 50,
              "max_accuracy": 0.9,
              "total_sessions": 100,
              "kdr": 2.52,
              "hs_rate": 0.6
            }'
    ```

