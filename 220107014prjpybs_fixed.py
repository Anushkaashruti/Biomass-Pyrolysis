#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split




# In[138]:


original_df=pd.read_csv("pyrolysis.csv")
anushka_df=pd.read_csv("pyrolysis.csv")


# In[139]:


anushka_df.shape


# In[140]:


anushka_df.info()


# In[141]:


columns_to_convert = ['PS', 'Solid phase', 'Liquid phase', 'Gas phase']
anushka_df[columns_to_convert] = anushka_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')


# In[142]:


anushka_df.dtypes


# In[143]:


duplicate_count=anushka_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

anushka_df.drop_duplicates()


# In[144]:


(anushka_df.isnull().sum())


# In[145]:


ax = sns.heatmap(anushka_df.isnull(), yticklabels=False, cbar=False,cmap="plasma")


# In[146]:


anushka_df.columns


# In[ ]:


columns_to_fill = [
    'M', 'Ash ', 'VM', 'FC', 'C', 'H', 'O', 'N',
       'PS', 'FT', 'HR', 'FR', 'Solid phase', 'Liquid phase', 'Gas phase'
]


valid_columns = [col for col in columns_to_fill if col in anushka_df.columns]

# Filling NaN values with mean
anushka_df[valid_columns] = anushka_df[valid_columns].fillna(anushka_df[valid_columns].mean())


# In[148]:


anushka_df.head


# In[149]:


ax = sns.heatmap(anushka_df.isnull(), yticklabels=False, cbar=False,cmap="plasma")


# In[150]:


anushka_df = anushka_df.apply(lambda col: col.astype('category').cat.codes if col.dtypes == 'object' else col)


# In[151]:


columns_to_drop = ['Unnamed: 0']

anushka_df = anushka_df.drop(columns=columns_to_drop, errors='ignore')


# In[ ]:


#correlation matrix
corr_matrix = anushka_df.corr()

# Plots
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='magma', linewidths=0.7, fmt=".2f", annot_kws={"size": 10},cbar_kws={"shrink": 0.8})

plt.title("Correlation Matrix Heatmap", fontsize=14)  
plt.xticks(fontsize=10, rotation=45)  
plt.yticks(fontsize=10)
plt.show()


# In[158]:


plt.figure(figsize=(14, 8))
sns.boxplot(data=anushka_df)
plt.title("Box Plot of Numeric Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[155]:


def remove_outliers_iqr(anushka_df):
    Q1 = anushka_df.quantile(0.25)
    Q3 = anushka_df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return anushka_df[~((anushka_df < lower_bound) | (anushka_df > upper_bound)).any(axis=1)]

# Remove outliers
anushka_df_clean = remove_outliers_iqr(anushka_df)

print("\nData after outlier removal:\n")
anushka_df_clean


# In[157]:


plt.figure(figsize=(14, 8))
sns.boxplot(data=anushka_df_clean)
plt.title("Box Plot of Numeric Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[156]:


plt.figure(figsize=(8, 6))
plt.scatter(anushka_df_clean['FT'], anushka_df_clean['Solid phase'], alpha=0.7, color='orange')
plt.title('Final Temperature vs. Solid Phase Yield')
plt.xlabel('Final Temperature (°C)')
plt.ylabel('Solid Phase Yield (%)')
plt.grid(True)
plt.show()


# In[ ]:


numeric_cols = anushka_df_clean.select_dtypes(include=['float64', 'int64'])

numeric_filled = numeric_cols.fillna(numeric_cols.mean())

# Apply PCA
pca = PCA(n_components=3)
from sklearn.preprocessing import StandardScaler
# Automatically selecting numeric features (adjust if needed)
X = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

pca_result = pca.fit_transform(scaled_data)

import matplotlib.pyplot as plt
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(range(1, 4), explained_variance, marker='o', linestyle='--', color='green')
plt.title('Scree Plot (Top 3 Principal Components)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()


# In[198]:


#linearregression
df = anushka_df_clean.drop(columns=['Index', 'Biomass species']).dropna()

# Split input and output
X = df.drop(columns=['Solid phase', 'Liquid phase', 'Gas phase'])
y = df[['Solid phase', 'Liquid phase', 'Gas phase']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# In[199]:


# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# In[200]:


# Evaluation function
def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n {model_name} Performance")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")

# Call the evaluation
evaluate_model(y_test, y_pred, "Linear Regression")


# In[ ]:


y_true_solid = y_test['Solid phase']
y_true_liquid = y_test['Liquid phase']
y_true_gas = y_test['Gas phase']

y_pred_solid = y_pred[:, 0]
y_pred_liquid = y_pred[:, 1]
y_pred_gas = y_pred[:, 2]

# Plot all 3 subplots
plt.figure(figsize=(18, 5))

# Solid Phase
plt.subplot(1, 3, 1)
plt.scatter(y_true_solid, y_pred_solid, color='royalblue', alpha=0.6)
plt.plot([y_true_solid.min(), y_true_solid.max()], [y_true_solid.min(), y_true_solid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Solid Phase')

# Liquid Phase
plt.subplot(1, 3, 2)
plt.scatter(y_true_liquid, y_pred_liquid, color='seagreen', alpha=0.6)
plt.plot([y_true_liquid.min(), y_true_liquid.max()], [y_true_liquid.min(), y_true_liquid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Liquid Phase')

# Gas Phase
plt.subplot(1, 3, 3)
plt.scatter(y_true_gas, y_pred_gas, color='darkorange', alpha=0.6)
plt.plot([y_true_gas.min(), y_true_gas.max()], [y_true_gas.min(), y_true_gas.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gas Phase')

plt.suptitle('Linear Regression: Actual vs Predicted Yields', fontsize=16)
plt.tight_layout()
plt.show()


# In[195]:


#Decisiontreeregression
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)

# Predict
y_pred = dt.predict(X_test_scaled)

# Evaluate
def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {model_name} Performance")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}

# Call it
metrics_dt = evaluate_model(y_test, y_pred, "Decision Tree Regressor")



# In[190]:


import matplotlib.pyplot as plt

# Predict using Decision Tree
y_pred_dt = dt.predict(X_test_scaled)

# Extract true values
y_true_solid = y_test['Solid phase']
y_true_liquid = y_test['Liquid phase']
y_true_gas = y_test['Gas phase']

# Extract predicted values
y_pred_solid = y_pred_dt[:, 0]
y_pred_liquid = y_pred_dt[:, 1]
y_pred_gas = y_pred_dt[:, 2]

# Plot
plt.figure(figsize=(18, 5))

# Solid Phase
plt.subplot(1, 3, 1)
plt.scatter(y_true_solid, y_pred_solid, color='royalblue', alpha=0.6)
plt.plot([y_true_solid.min(), y_true_solid.max()], [y_true_solid.min(), y_true_solid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Solid Phase')

# Liquid Phase
plt.subplot(1, 3, 2)
plt.scatter(y_true_liquid, y_pred_liquid, color='seagreen', alpha=0.6)
plt.plot([y_true_liquid.min(), y_true_liquid.max()], [y_true_liquid.min(), y_true_liquid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Liquid Phase')

# Gas Phase
plt.subplot(1, 3, 3)
plt.scatter(y_true_gas, y_pred_gas, color='darkorange', alpha=0.6)
plt.plot([y_true_gas.min(), y_true_gas.max()], [y_true_gas.min(), y_true_gas.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gas Phase')

plt.suptitle('Decision Tree Regressor: Actual vs Predicted Yields', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = rf.predict(X_test_scaled)


# In[176]:


# Evaluation function
def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
metrics_rf = evaluate_model(y_test, y_pred, "Random Forest Regressor")


# In[191]:


import matplotlib.pyplot as plt

# Predict using Random Forest
y_pred_rf = rf.predict(X_test_scaled)

# Extract true values
y_true_solid = y_test['Solid phase']
y_true_liquid = y_test['Liquid phase']
y_true_gas = y_test['Gas phase']

# Extract predicted values
y_pred_solid = y_pred_rf[:, 0]
y_pred_liquid = y_pred_rf[:, 1]
y_pred_gas = y_pred_rf[:, 2]

# Plot
plt.figure(figsize=(18, 5))

# Solid Phase
plt.subplot(1, 3, 1)
plt.scatter(y_true_solid, y_pred_solid, color='royalblue', alpha=0.6)
plt.plot([y_true_solid.min(), y_true_solid.max()], [y_true_solid.min(), y_true_solid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Solid Phase')

# Liquid Phase
plt.subplot(1, 3, 2)
plt.scatter(y_true_liquid, y_pred_liquid, color='seagreen', alpha=0.6)
plt.plot([y_true_liquid.min(), y_true_liquid.max()], [y_true_liquid.min(), y_true_liquid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Liquid Phase')

# Gas Phase
plt.subplot(1, 3, 3)
plt.scatter(y_true_gas, y_pred_gas, color='darkorange', alpha=0.6)
plt.plot([y_true_gas.min(), y_true_gas.max()], [y_true_gas.min(), y_true_gas.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gas Phase')

plt.suptitle('Random Forest Regressor: Actual vs Predicted Yields', fontsize=16)
plt.tight_layout()
plt.show()


# In[179]:


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# In[180]:


# Predict
y_pred = knn.predict(X_test_scaled)


# In[181]:


# Evaluation function
def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
metrics_knn = evaluate_model(y_test, y_pred, "KNN Regressor")


# In[194]:


import matplotlib.pyplot as plt

# Predict using KNN
y_pred_knn = knn.predict(X_test_scaled)

# Extract true values
y_true_solid = y_test['Solid phase']
y_true_liquid = y_test['Liquid phase']
y_true_gas = y_test['Gas phase']

y_pred_solid = y_pred_knn[:, 0]
y_pred_liquid = y_pred_knn[:, 1]
y_pred_gas = y_pred_knn[:, 2]

# Plot
plt.figure(figsize=(18, 5))

# Solid Phase
plt.subplot(1, 3, 1)
plt.scatter(y_true_solid, y_pred_solid, color='royalblue', alpha=0.6)
plt.plot([y_true_solid.min(), y_true_solid.max()], [y_true_solid.min(), y_true_solid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Solid Phase')

# Liquid Phase
plt.subplot(1, 3, 2)
plt.scatter(y_true_liquid, y_pred_liquid, color='seagreen', alpha=0.6)
plt.plot([y_true_liquid.min(), y_true_liquid.max()], [y_true_liquid.min(), y_true_liquid.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Liquid Phase')

# Gas Phase
plt.subplot(1, 3, 3)
plt.scatter(y_true_gas, y_pred_gas, color='darkorange', alpha=0.6)
plt.plot([y_true_gas.min(), y_true_gas.max()], [y_true_gas.min(), y_true_gas.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gas Phase')

plt.suptitle('KNN Regressor: Actual vs Predicted Yields', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd


model_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'KNN'],
    'R²':     [0.299, 0.627, 0.726, 0.554],
    'RMSE':   [6.78,  4.37,  3.78,  5.08],
    'MAE':    [4.95,  2.51,  2.28,  3.37],
    'MSE':    [45.94, 19.14, 14.29, 25.80]
})

model_results = model_results.sort_values(by='R²', ascending=False).reset_index(drop=True)

print("📊 Solid Phase – Model Performance Comparison")
display(model_results)


#  ## **Model Performance Summary: Phase-wise Yield Prediction**
# 
# **1. Linear Regression**
# - A simple baseline model assuming linear relationships.
# - **R² Score:** Lowest across all phases (e.g., 0.299 for Solid phase).
# - **Error:** Highest RMSE and MAE among all models.
# - **Observation:** Fails to capture the non-linearity present in biomass pyrolysis behavior.
# 
# **2. Decision Tree Regressor**
# - Non-linear, rule-based model that splits data based on conditions.
# - **R² Score:** Moderate (e.g., 0.627 for Solid phase).
# - **Error:** Improved over Linear Regression but susceptible to overfitting.
# - **Observation:** Captures patterns better than linear models but lacks ensemble stability.
# 
# **3. K-Nearest Neighbors (KNN)**
# - Instance-based learning that predicts using nearby data points.
# - **R² Score:** Moderate to low (e.g., 0.554 for Solid phase).
# - **Error:** Higher RMSE and MAE than Decision Tree.
# - **Observation:** Performance depends heavily on feature scaling and data distribution.
# 
# **4. Random Forest Regressor**
# - Ensemble model that aggregates multiple Decision Trees.
# - **R² Score:** Highest across all phases (e.g., 0.726 for Solid phase).
# - **Error:** Lowest RMSE and MAE values.
# - **Observation:** Best at capturing complex, non-linear patterns; highly stable and accurate.
# 
# ### **Conclusion**
# - Among all the models evaluated, the **Random Forest Regressor** consistently delivered the most accurate predictions for **solid, liquid, and gas phase yields**.
# - Its ensemble learning approach helps in reducing overfitting while handling feature interactions effectively.
# - In contrast, **Linear Regression** lacked the flexibility to model the complexity of pyrolysis behavior, and while **Decision Tree** and **KNN** performed better, they were still limited in stability and generalization.
# 
#  **Therefore, Random Forest is identified as the most suitable model** for predicting phase-wise product yields in biomass pyrolysis, making it a reliable tool for process optimization in renewable energy and biofuel applications.

# 
