# save_feature_names.py  — run once, takes 2 seconds
import pandas as pd
import pickle
import os

DATASET_DIR = r'D:\base\6g-iot-security\data'
RESULTS_DIR = r'D:\base\mullti\ids_results'

# Read only the header row — no data loaded at all
csv_files = sorted([os.path.join(DATASET_DIR, f)
                    for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])

# Read just 1 row to get column names
df_header = pd.read_csv(csv_files[0], nrows=1)
feature_names = df_header.columns[:-1].tolist()

print(f"Found {len(feature_names)} features")
print(f"First 5 : {feature_names[:5]}")
print(f"Last  5 : {feature_names[-5:]}")

# Save
with open(os.path.join(RESULTS_DIR, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feature_names, f)

print(f"\nSaved → {RESULTS_DIR}\\feature_names.pkl")
print("Now run shap_explain.py")