import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import shap
from sklearn.tree import DecisionTreeClassifier

# dataset loaded
df = pd.read_csv('mxmh_survey_results.csv')
#----------------------------------Handling missing values of Age feature -----------------------------------------------------------------------------
# Selecting relevant numeric columns for imputation
numeric_cols = ['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression', 'Insomnia', 'OCD']
df_numeric = df[numeric_cols]

# Initializing KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Performing KNN imputation
imputed_data = imputer.fit_transform(df_numeric)

print("Imputed data based on prediction\n",imputed_data)
print("\n")
# Creating a new dataframe with imputed values
df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)


# Replacing the original 'Age' column with the imputed values
df['Age'] = df_imputed['Age']
df['Age'] = df['Age'].astype(int)


#-----------------------------------------Handling missing values of categorical features---------------------------------------------------------
# Filling missing values using forward fill
df['Primary streaming service'] = df['Primary streaming service'].ffill()
df['While working']=df['While working'].ffill()
df['Instrumentalist'] =df['Instrumentalist'].ffill()
df['Composer']=df['Composer'].ffill()
df['Composer']=df['Composer'].ffill()

#--------------------------------------Handling missing value of BPM feature-------------------------------------------------------------------
## 1. Spliting data into rows with and without missing values
train_data = df[df['BPM'].notnull()]
missing_data = df[df['BPM'].isnull()]

#Features used to predict values of BPM
categorical_columns = ['Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]',
'Frequency [Folk]', 'Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]',
'Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]',
'Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]'
]

# One-hot encoding categorical columns for training and missing data
train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
missing_data_encoded = pd.get_dummies(missing_data, columns=categorical_columns, drop_first=True)

# Align columns in missing_data_encoded to match the columns in train_data_encoded
missing_data_encoded = missing_data_encoded.reindex(columns=train_data_encoded.columns, fill_value=0)

# Features which have less correlation with feature BPM
col = ['Timestamp','Age','Primary streaming service','Hours per day',
'While working','Instrumentalist','Composer',
'Fav genre','Exploratory','Foreign languages','BPM','Anxiety',
'Depression','Insomnia','OCD','Music effects']

# Splitting into features and target
X_train = train_data_encoded.drop(col, axis=1) # Features for training
y_train = train_data_encoded['BPM'] #Target (BPM)
X_missing = missing_data_encoded.drop(col, axis=1) # Features for missing data

# Train a RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the missing BPM values
predicted_values = model.predict(X_missing)

# Fill missing values in the original dataframe
df.loc[df['BPM'].isnull(), 'BPM'] = predicted_values
#-----------------------------------Handling missing value of Music effects feature--------------------------------------------------------------
# Fill missing values temporarily with 'Unknown' for encoding
df['Music effects']= df['Music effects'].fillna('Unknown')
 
# Encode the 'Music effects' column
label_encoder = LabelEncoder()
df['Music effects'] = label_encoder.fit_transform(df['Music effects'])

# Select features for prediction (you can add more relevant features)
features = ['Age', 'Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD']

# Prepare the data for training
known_data = df[df['Music effects'] != label_encoder.transform(['Unknown'])[0]]
unknown_data = df[df['Music effects'] == label_encoder.transform(['Unknown'])[0]]

X_known = known_data[features]
y_known = known_data['Music effects']
X_unknown = unknown_data[features]

# Train a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict missing 'Music effects' values if X_unknown is not empty
if not X_unknown.empty:
    y_pred_unknown = rf_model.predict(X_unknown)
    df.loc[df['Music effects'] == label_encoder.transform(['Unknown'])[0], 'Music effects'] = y_pred_unknown

# Decode the 'Music effects' column back to original labels
df['Music effects'] = label_encoder.inverse_transform(df['Music effects'])

#----------------------------------------removing noise and scaling numerical features-------------------------------------------------------------------------
scaler = MinMaxScaler()

# Removes rows where BPM is above 200
df = df[df['BPM'] < 250]  # Removes rows where BPM is above 200

# Scale features using Normalization
numeric_cols = ['BPM', 'Hours per day']
df.loc[:, numeric_cols] = scaler.fit_transform(df.loc[:, numeric_cols])
#----------------------------------------------Encoding categorical features-----------------------------------------------------


categorical_cols = ['Primary streaming service', 'While working','Instrumentalist','Composer','Fav genre'
,'Exploratory','Foreign languages','Frequency [Classical]','Frequency [Country]', 'Frequency [EDM]',
'Frequency [Folk]', 'Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]',
'Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]',
'Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]','Music effects']
numerical_cols = ['Age', 'Hours per day','BPM','Anxiety', 'Depression', 'Insomnia', 'OCD']



EncodedDF = pd.get_dummies(df, columns=categorical_cols)
EncodedDF.drop(columns=['Timestamp'], inplace=True)

#---------------------------Decision tree model(feature selection)----------------------------------------------------------
dropFeatures = ['Music effects_Improve','Music effects_No effect','Music effects_Worsen']
X = EncodedDF.drop(columns=dropFeatures)
y = EncodedDF[dropFeatures] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt_model.fit(X_train, y_train)

importances = dt_model.feature_importances_

# Create a DataFrame to view feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
     'Coefficient': importances
}).sort_values(by='Coefficient', ascending=False)

print("Selected features based on decision tree\n\n",feature_importance_df['Feature'].head(10).tolist())

#--------------------------------------------Linear Regression----------------------------------
X_selected=X[feature_importance_df['Feature'].head(10)]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("**********Linear Regression*********\n")
print("Root Mean Squared Error (MSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))
print("\n")
#---------------------------Cross-validation------------------------------------------------------

rf_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_model, X_selected, y, cv=5, scoring='accuracy')
print("**********Cross Validation*********\n")
# Print out the accuracy for each fold
print("Cross-validation scores for each fold:", cv_scores)

# Compute the average accuracy across all folds
average_accuracy = cv_scores.mean()
print("**********Cross fold validation*********\n")
print(f"Average accuracy across all folds: {average_accuracy:.4f}")
print("\n")
#--------------------------------------feature importance heat map---------------------------------------------------------

coefficients_df = pd.DataFrame(model.coef_, columns=X_selected.columns)
coefficients_df.index = dropFeatures

plt.figure(figsize=(10, 6))
sns.heatmap(coefficients_df, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Importance for Multiple Targets")
plt.xlabel("Features")
plt.ylabel("Target Variables")
plt.show()
#-------------------------------residual plot----------------------------------------------------------
# Predictions
y_pred = model.predict(X_test)

# Calculate residuals for each target variable
residuals_improve = y_test['Music effects_Improve'] - y_pred[:, 0]
residuals_no_effect = y_test['Music effects_No effect'] - y_pred[:, 1]
residuals_worsen = y_test['Music effects_Worsen'] - y_pred[:, 2]

# Plotting residuals for each target variable
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Residual Plot for 'Music effects_Improve'
sns.scatterplot(x=y_pred[:, 0], y=residuals_improve, ax=axs[0])
axs[0].axhline(y=0, color='r', linestyle='--')
axs[0].set_title("Residual Plot for 'Music effects_Improve'")
axs[0].set_xlabel("Predicted Values")
axs[0].set_ylabel("Residuals")

# Residual Plot for 'Music effects_No effect'
sns.scatterplot(x=y_pred[:, 1], y=residuals_no_effect, ax=axs[1])
axs[1].axhline(y=0, color='r', linestyle='--')
axs[1].set_title("Residual Plot for 'Music effects_No effect'")
axs[1].set_xlabel("Predicted Values")
axs[1].set_ylabel("Residuals")

# Residual Plot for 'Music effects_Worsen'
sns.scatterplot(x=y_pred[:, 2], y=residuals_worsen, ax=axs[2])
axs[2].axhline(y=0, color='r', linestyle='--')
axs[2].set_title("Residual Plot for 'Music effects_Worsen'")
axs[2].set_xlabel("Predicted Values")
axs[2].set_ylabel("Residuals")

# Adjust layout and show plots
plt.tight_layout()
plt.show()

print("*********Residual plots on categorical target*********\n")
print("Root Mean Squared Error (RMSE) for 'Music effects_Improve':", np.sqrt(mean_squared_error(y_test['Music effects_Improve'], y_pred[:, 0])))
print("Root Mean Squared Error (RMSE) for 'Music effects_No effect':", np.sqrt(mean_squared_error(y_test['Music effects_No effect'], y_pred[:, 1])))
print("Root Mean Squared Error (RMSE) for 'Music effects_Worsen':", np.sqrt(mean_squared_error(y_test['Music effects_Worsen'], y_pred[:, 2])))
print("\n")
#------------------------------------------l2 Regularization------------------------------------------------------
ridge_model = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
print("**********l2 Regularization*********\n")
print("Ridge Model - Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("Ridge Model - R^2 Score:", r2_score(y_test, y_pred_ridge))
print("\n")

#--------------------------------------------------l1 Regularization-----------------------------------------------------
lasso_model = Lasso(alpha=0.01, positive=True)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("**********l1 Regularization*********\n")
print(f"Lasso with Lower Alpha - RMSE: {rmse_lasso}, R^2: {r2_lasso}")
print("\n")
#-------------------------------------Polynomial Regression------------------------------
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

X_test_poly = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)
print("**********Polynomial Regression*********\n")
print(f"Polynomial Regression - RMSE: {rmse_poly}, R^2: {r2_poly}")
print("\n")

#---------------------------------------------Shap package for reporting-------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer(X_test)
sample_idx = 0  # Index of the sample to explain (change as needed)
for i, target in enumerate(['Music effects_Improve', 'Music effects_No effect', 'Music effects_Worsen']):
    print(f"Saving SHAP Force Plot for Target: {target}")
    force_plot = shap.force_plot(
        explainer.expected_value[i], 
        shap_values.values[sample_idx, :, i],  
        X_test.iloc[sample_idx],  
        feature_names=X_test.columns
    )
    
    shap.save_html(f"force_plot_target_{i}.html", force_plot)
    print(f"Force plot for {target} saved as force_plot_target_{i}.html")

