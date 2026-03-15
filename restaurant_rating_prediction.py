import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the restaurant dataset.
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Drop columns that are not useful for prediction or cause leakage
    # Rating color and Rating text are based on Aggregate rating, so they are leaky.
    # Restaurant ID, Name, Address, etc. are identifiers.
    # City might be useful but high cardinality - encoding it or dropping for simplicity. 
    # Let's keep City and Label Encode it for now to see if location matters.
    drop_cols = ['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Rating color', 'Rating text', 'Switch to order menu']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    print(f"Columns after dropping irrelevant features: {df.columns.tolist()}")

    # Handle missing values
    # 'Cuisines' usually has some missing values
    if df.isnull().sum().any():
        print("Handling missing values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna() # For simplicity, dropping rows with missing values as dataset is large enough
        
    # Feature Engineering
    # 'Cuisines' is a list of cuisnes. We can create a feature for Number of Cuisines
    df['Number_of_Cuisines'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))
    
    # Encode Categorical Variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("Encoding categorical variables...")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    return df

def train_and_evaluate(df):
    """
    Trains models and evaluates them.
    """
    X = df.drop(columns=['Aggregate rating'])
    y = df['Aggregate rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MSE": mse, "R2": r2, "Model": model}
        
        print(f"{name} Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        with open("model_metrics.txt", "a") as f:
            f.write(f"\n{name} Results:\n")
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"R-squared: {r2:.4f}\n")


    return results, X_train.columns

def analyze_feature_importance(model, feature_names):
    """
    Analyzes feature importance for Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
    # Plotting (Optional - saving to file implies we might want to view it, but simple print is fine for now)
    # plt.figure(figsize=(10, 6))
    # plt.title("Feature Importances")
    # plt.bar(range(len(feature_names)), importances[indices], align="center")
    # plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    # plt.tight_layout()
    # plt.savefig('feature_importance.png')

if __name__ == "__main__":
    dataset_path = 'c:/Users/drash/OneDrive/Desktop/project/Cognifyz Project/Dataset .csv'
    
    # Preprocess
    df_processed = load_and_preprocess_data(dataset_path)
    
    # Train
    model_results, feature_columns = train_and_evaluate(df_processed)
    
    # Analyze Random Forest
    rf_model = model_results["Random Forest"]["Model"]
    analyze_feature_importance(rf_model, feature_columns)
