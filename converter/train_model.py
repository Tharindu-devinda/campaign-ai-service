"""
Train and export Explainable AI Decision Tree Model for Campaign Failure Diagnosis.
Implements Section 5.5 of Multi-Tenant Campaign Assistant Specification:
'Explainable model inside the service, loaded from model.joblib at startup. A few MB, CPU only, milliseconds per prediction.'
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

FEATURE_NAMES = [
    'total_rows',
    'valid_rows',
    'duplicate_count',
    'duplicate_ratio',
    'invalid_count',
    'invalid_ratio',
    'empty_rows',
    'empty_rows_ratio',
    'country_mismatch_ratio',
    'file_missing',
    'log_missing',
    'log_total_lines',
    'error_count',
    'error_ratio',
    'rate_limit_count',
    'rate_limit_ratio',
    'rate_limit_burst_max',
    'gateway_timeout_count',
    'gateway_timeout_ratio',
    'auth_failed_count',
    'auth_failed_ratio',
    'low_balance_count',
    'low_balance_ratio',
    'invalid_dest_count',
    'invalid_dest_ratio',
    'network_drop_count',
    'first_error_pos_ratio',
    'failed_at_start',
    'failed_midway',
    'send_speed_per_sec',
    'message_char_length',
    'delivery_success_ratio',
    'completion_ratio',
]

CLASSES = [
    'NORMAL',
    'FILE_PROBLEM',
    'RATE_LIMIT_EXCEEDED',
    'GATEWAY_TIMEOUT',
    'AUTH_OR_BALANCE',
    'INVALID_DATA'
]

def generate_synthetic_training_data(n_samples: int = 1200):
    np.random.seed(42)
    rows = []
    labels = []

    for _ in range(n_samples):
        # Baseline normal campaign
        total_rows = np.random.randint(100, 100000)
        dup_ratio = np.random.uniform(0.0, 0.04)
        inv_ratio = np.random.uniform(0.0, 0.02)
        empty_ratio = np.random.uniform(0.0, 0.01)
        mismatch_ratio = np.random.uniform(0.0, 0.05)
        rate_limit_cnt = 0
        gw_timeout_cnt = 0
        auth_cnt = 0
        balance_cnt = 0
        speed = np.random.uniform(10.0, 30.0)
        failed_start = 0
        failed_midway = 0

        # Inject failure scenarios
        scenario = np.random.choice(CLASSES, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10])

        if scenario == 'FILE_PROBLEM':
            # High duplicates, invalid formats or empty rows
            choice = np.random.choice(['dup', 'inv', 'empty', 'mixed'])
            if choice == 'dup':
                dup_ratio = np.random.uniform(0.08, 0.45)
            elif choice == 'inv':
                inv_ratio = np.random.uniform(0.05, 0.25)
            elif choice == 'empty':
                empty_ratio = np.random.uniform(0.05, 0.30)
            else:
                dup_ratio = np.random.uniform(0.06, 0.20)
                inv_ratio = np.random.uniform(0.04, 0.15)
        elif scenario == 'RATE_LIMIT_EXCEEDED':
            speed = np.random.uniform(40.0, 120.0)
            rate_limit_cnt = np.random.randint(8, 150)
        elif scenario == 'GATEWAY_TIMEOUT':
            gw_timeout_cnt = np.random.randint(2, 40)
            failed_start = np.random.choice([0, 1])
        elif scenario == 'AUTH_OR_BALANCE':
            if np.random.rand() > 0.5:
                auth_cnt = np.random.randint(1, 10)
                failed_start = 1
            else:
                balance_cnt = np.random.randint(1, 5)
                failed_midway = 1
        elif scenario == 'INVALID_DATA':
            mismatch_ratio = np.random.uniform(0.25, 0.85)

        dup_cnt = int(total_rows * dup_ratio)
        inv_cnt = int(total_rows * inv_ratio)
        empty_cnt = int(total_rows * empty_ratio)
        valid_rows = max(0, total_rows - dup_cnt - inv_cnt - empty_cnt)

        feat_vector = {
            'total_rows': total_rows,
            'valid_rows': valid_rows,
            'duplicate_count': dup_cnt,
            'duplicate_ratio': dup_ratio,
            'invalid_count': inv_cnt,
            'invalid_ratio': inv_ratio,
            'empty_rows': empty_cnt,
            'empty_rows_ratio': empty_ratio,
            'country_mismatch_ratio': mismatch_ratio,
            'file_missing': 0.0,
            'log_missing': 0.0 if scenario not in ['NORMAL', 'FILE_PROBLEM'] else 1.0,
            'log_total_lines': 50.0,
            'error_count': rate_limit_cnt + gw_timeout_cnt + auth_cnt + balance_cnt,
            'error_ratio': (rate_limit_cnt + gw_timeout_cnt) / max(1, total_rows),
            'rate_limit_count': rate_limit_cnt,
            'rate_limit_ratio': rate_limit_cnt / max(1, total_rows),
            'rate_limit_burst_max': min(25, rate_limit_cnt),
            'gateway_timeout_count': gw_timeout_cnt,
            'gateway_timeout_ratio': gw_timeout_cnt / max(1, total_rows),
            'auth_failed_count': auth_cnt,
            'auth_failed_ratio': 1.0 if auth_cnt > 0 else 0.0,
            'low_balance_count': balance_cnt,
            'low_balance_ratio': 1.0 if balance_cnt > 0 else 0.0,
            'invalid_dest_count': 0.0,
            'invalid_dest_ratio': 0.0,
            'network_drop_count': 0.0,
            'first_error_pos_ratio': 0.05 if failed_start else 0.5,
            'failed_at_start': float(failed_start),
            'failed_midway': float(failed_midway),
            'send_speed_per_sec': speed,
            'message_char_length': 45.0,
            'delivery_success_ratio': 0.95 if scenario == 'NORMAL' else 0.2,
            'completion_ratio': 1.0 if scenario == 'NORMAL' else 0.3,
        }

        row = [feat_vector[k] for k in FEATURE_NAMES]
        rows.append(row)
        labels.append(scenario)

    return pd.DataFrame(rows, columns=FEATURE_NAMES), np.array(labels)

def train_and_export():
    print("Generating synthetic training dataset from multi-tenant baseline rules...")
    X, y = generate_synthetic_training_data(2500)

    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    print(f"Decision Tree trained with {accuracy*100:.2f}% accuracy.")

    out_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    model_payload = {
        'model': clf,
        'feature_names': FEATURE_NAMES,
        'classes': clf.classes_.tolist(),
        'version': '1.0.0'
    }
    joblib.dump(model_payload, out_path)
    print(f"Exported Explainable AI model to: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == '__main__':
    train_and_export()
