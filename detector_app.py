import joblib
import pandas as pd
import numpy as np

def load_dataset(filename: str):
    try:
        df = pd.read_csv(filename)
        df = df.drop_duplicates()
        df["Amount_log"] = np.log1p(df["Amount"])
        df = df.drop(columns=["Amount"])
        return df
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    model = joblib.load("models/xgboost_fraud.pkl")
    df = load_dataset("data/creditcard.csv")

    caught = 0
    missed = 0
    false_alarms = 0
    correct_normals = 0

    for i in range(20):
        random_row = df.sample(n=1)
        is_fraud = random_row["Class"].values[0] == 1
        predict = model.predict(random_row.drop(columns=["Class"]))[0] == 1
        
        if is_fraud and predict:
            print(f"{i+1}: FRAUD CAUGHT")
            caught += 1
        elif is_fraud and not predict:
            print(f"{i+1}: FRAUD MISSED")
            missed += 1
        elif not is_fraud and predict:
            print(f"{i+1}: FALSE ALARM")
            false_alarms += 1
        else:
            print(f"{i+1}: NORMAL")
            correct_normals += 1

    print(f"Caught: {caught}")
    print(f"Missed: {missed}")
    print(f"False alarms: {false_alarms}")
    print(f"Normal: {correct_normals}")