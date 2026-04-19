import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load Data
df = pd.read_csv("Bengaluru_House_Data.csv")

# 2. Clean total_sqft (Use the function we wrote earlier)
def convert_sqft_to_num(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# 3. Create BHK Column (CRITICAL: Do this BEFORE defining X)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)

# 4. Handle Missing Values
df = df.dropna(subset=['total_sqft', 'bhk', 'price'])

# 5. Define Features (Now 'bhk' exists!)
X = df[['total_sqft', 'bhk']]
y = df['price']

# 6. Train-Test Split and Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# 7. Save the model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(lr_clf, f)

print("Model trained and saved successfully!")
