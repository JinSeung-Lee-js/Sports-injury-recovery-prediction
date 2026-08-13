# Sports-injury-recovery-prediction
ML pipeline diagnosing predictive signal loss (R² ≈ -0.05) and validating via pharmacologically-grounded synthetic data (R² = 0.41)
# Sports Injury Recovery Time Prediction

## Question
Can an athlete's recovery time from a sports injury be predicted from
clinical/biometric data and treatment type — and if medication (NSAIDs/
steroids) isn't in the data, does adding it change what the model can learn?

## Data
- **Stage 1** (`Athlete_recovery_dataset.csv`): 1,000 real athlete records
  — injury type/severity, therapy type, heart rate, blood pressure, sleep,
  training load, etc. No medication variable available.
- **Stage 2** (`synthetic_drug_recovery_dataset.csv`): a synthetic dataset
  I generated with pharmacologically-grounded assumptions (NSAID/steroid
  effects on recovery time), used to re-validate the same pipeline.

## Approach
Same regression pipeline (Linear Regression, Random Forest) applied to
both datasets, so any difference in results reflects the data itself,
not the method.

## Key Finding: Diagnosing "No Signal" vs. Fixing the Pipeline
On the real dataset, the model scored **R² ≈ -0.05** — worse than just
predicting the average. A correlation heatmap confirmed why: every
variable had a correlation coefficient under 0.1 with recovery time.
This wasn't a modeling failure; the available variables simply didn't
carry a predictable signal.

To check whether the *pipeline* itself was sound, I built a synthetic
dataset with known, pharmacologically-motivated relationships (steroid/
NSAID effects, injury severity, sleep) and reran the same models. Result:
**R² improved to 0.37–0.41**, and `Medication_Steroid` showed up as a
meaningful feature — confirming the pipeline works correctly when the
data actually contains a signal to learn.

## Tech Stack
Python, pandas, scikit-learn (Linear Regression, Random Forest, Logistic
Regression), matplotlib/seaborn

## How to Run
​```
pip install pandas numpy scikit-learn matplotlib seaborn
​```
Open `01_athlete_recovery_analysis.ipynb` and `02_synthetic_drug_recovery_analysis.ipynb`
and run all cells (datasets are loaded from the same folder).

## Full Notebooks
- [`01_athlete_recovery_analysis.ipynb`](./01_athlete_recovery_analysis.ipynb) — real-data analysis
- [`02_synthetic_drug_recovery_analysis.ipynb`](./02_synthetic_drug_recovery_analysis.ipynb) — synthetic-data validation
