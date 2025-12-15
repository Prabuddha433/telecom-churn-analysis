import pandas as pd
import numpy as np

def feature_engineering(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges
    df['TotalCharges'] = df['TotalCharges'].fillna(0)


    # Average charge per month
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 60, 100],
        labels=["0-6", "6-12", "12-24", "24-60", "60+"]
    )

    # Long-term customer flag
    df["IsLongTermCustomer"] = np.where(df["tenure"] >= 12, 1, 0)

    # Count number of services subscribed
    services = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["TotalServicesSubscribed"] = df[services].apply(
        lambda x: (x == "Yes").sum(), axis=1
    )

    return df
