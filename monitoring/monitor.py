import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load dataset
file_path = "data/raw/german.data-numeric"

df = pd.read_csv(file_path, sep=r"\s+", header=None)

# Last column is target
X = df.drop(columns=[24])
X.columns = X.columns.astype(str)

# Split data
reference_data, current_data = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

# Create Evidently report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Run data drift analysis
report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save HTML report
report.save_html("monitoring/data_drift_report.html")

print("Evidently monitoring report generated successfully!")