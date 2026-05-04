# Crime Category Classification

This is an end-to-end Machine Learning project to perform Crime Category Classification.

## Structure
- `data/` : Add your `train.csv` and other datasets here. (These are ignored by Git).
- `notebooks/` : Contains the main workflow Jupyter Notebook.
- `src/` : Reusable modular Python scripts for data processing, modeling, optimization, and evaluation.
- `models/` : Saved trained models.
- `outputs/` : Generated plots and CSV outputs.

## Setup Instructions

1. **Activate Virtual Environment:**
   On Windows:
   `venv\Scripts\Activate.ps1`
   On Linux/Mac:
   `source venv/bin/activate`

2. **Install Dependencies:**
   `pip install -r requirements.txt`

## Running the Project
1. Place a dataset in `data/train.csv`.
2. Navigate to the `notebooks/` folder or start Jupyter Notebook: `jupyter notebook`
3. Open `crime_classification.ipynb` and run all cells.

## Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Neural Network (TensorFlow)