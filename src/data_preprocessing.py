import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset successfully loaded from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please add the dataset.")
        return None

def clean_data(df):
    print("Initial missing values:")
    print(df.isnull().sum())
    
    leakage_cols = ['Descript', 'Description', 'Resolution']
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped data leakage columns: {cols_to_drop}")
        print(f"Final list of features (including target): {list(df.columns)}")
    
    # Fill missing values with median for numerical and mode for categorical
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    return df

def encode_features(df, target_col):
    encoders = {}
    df_encoded = df.copy()
    
    # Encode conceptual object columns
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    print(f"Final list of features (X): {list(X.columns)}")
    
    return X, y, encoders

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
