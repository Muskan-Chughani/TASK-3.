# TASK-3.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LassoCV

# Load the dataset from the provided path
file_path = '/content/drive/MyDrive/internship/data.csv'
data = pd.read_csv(file_path)

# Display the column names and data types
print(data.info())

# Handle missing values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Separate the features and target variable
target_column = 'fail'  # Replace with the actual target column name if different
X = data.drop(target_column, axis=1)
y = data[target_column]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the KNN model without feature selection
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_no_fs = accuracy_score(y_test, y_pred)
print(f'Accuracy without feature selection: {accuracy_no_fs}')

# Define the number of top features to select
num_features = min(10, X_train.shape[1])

# Filter Method: Mutual Information
mi = mutual_info_classif(X_train, y_train)
mi_threshold = sorted(mi, reverse=True)[:num_features][-1]
X_train_mi = X_train[:, mi >= mi_threshold]
X_test_mi = X_test[:, mi >= mi_threshold]

# Wrapper Method: Recursive Feature Elimination (RFE)
rfe = RFE(estimator=KNeighborsClassifier(n_neighbors=5), n_features_to_select=num_features)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Embedded Method: Lasso Regression
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
lasso_threshold = sorted(abs(lasso_coef), reverse=True)[:num_features][-1]
X_train_lasso = X_train[:, abs(lasso_coef) >= lasso_threshold]
X_test_lasso = X_test[:, abs(lasso_coef) >= lasso_threshold]


knn.fit(X_train_mi, y_train)
y_pred_mi = knn.predict(X_test_mi)
accuracy_mi = accuracy_score(y_test, y_pred_mi)
print(f'Accuracy with Mutual Information: {accuracy_mi}')


knn.fit(X_train_rfe, y_train)
y_pred_rfe = knn.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f'Accuracy with RFE: {accuracy_rfe}')


knn.fit(X_train_lasso, y_train)
y_pred_lasso = knn.predict(X_test_lasso)
accuracy_lasso = accuracy_score(y_test, y_pred_lasso)
print(f'Accuracy with Lasso: {accuracy_lasso}')


results = {
    'No Feature Selection': accuracy_no_fs,
    'Mutual Information': accuracy_mi,
    'RFE': accuracy_rfe,
    'Lasso': accuracy_lasso
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df)
