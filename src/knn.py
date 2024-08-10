import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
svm = SVC()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# df = pd.read_csv('NEO_asteroids.csv', parse_dates=True, header=0)
df = pd.read_csv('data/sbdb_query_results.csv', parse_dates=True, header=0)
#FILL PHA NAN VALUES WITH MODE VALUE
print(df)
df.fillna(df.mean(), inplace=True)
df.drop(['M1','full_name', 'pdes','name', 'M2', 'K1','K2', 'PC','extent','GM','IR','spec_B','spec_T', 'H_sigma', 'diameter_sigma', 'epoch', 'epoch_mjd','epoch_cal', 'equinox','two_body', 'A1', 'A1_sigma','A2', 'A2_sigma', 'A3', 'A3_sigma', 'DT', 'DT_sigma'], axis=1, inplace=True)
print(df)
df.fillna(0, inplace=True)
df['pha'] = df['pha'].map({'Y': 1, 'N': 0})
nan_count = df['pha'].isnull().sum()
print(f"Number of NaN values in 'pha': {nan_count}")
if nan_count > 0:
    df['pha'].fillna(df['pha'].mode()[0], inplace=True)
    print("Filled NaN values in 'pha' with mode.")
nan_count_after = df['pha'].isnull().sum()
print(f"Number of NaN values in 'pha' after handling: {nan_count_after}")



numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

nan_count_total = df.isnull().sum().sum()
print(f"Number of NaN values in the DataFrame after handling: {nan_count_total}")

if nan_count_total > 0:
    print("Columns with remaining NaN values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
   
   
columns_to_drop = ['pha',  'orbit_id', 'prefix'] 
X = df.drop(columns=columns_to_drop)
print(X)
y = df['pha']
df['pha'] = df['pha'].astype(float)
assert 'full_name' not in X.columns, "'full_name' is still in the features!"
assert not X.isnull().any().any(), "Missing values found in features."
print("Number of NaN values in target variable 'y' before splitting:", y.isnull().sum())
print("Columns in X after dropping specified columns:")
print(X.columns)
for col in X.select_dtypes(include=['object']).columns:
    print(f"Encoding column: {col}")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

for col in X.select_dtypes(include=['object']).columns:
    print(f"Encoding column: {col}")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("Data types in X after conversion:")
print(X.dtypes)

assert not X.isnull().any().any(), "Missing values found in features."
print("Number of NaN values in target variable 'y' before splitting:", y.isnull().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Number of NaN values in 'y_train':", y_train.isnull().sum())
print("Number of NaN values in 'y_test':", y_test.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




from sklearn.inspection import permutation_importance
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Precision:", precision_score(y_test, y_pred_knn))
print("KNN Recall:", recall_score(y_test, y_pred_knn))
print("KNN F1 Score:", f1_score(y_test, y_pred_knn))
print("KNN R² value:", r2_score(y_test, y_pred_knn))
print("KNN RMSE:", mean_squared_error(y_test, y_pred_knn, squared=False))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
cm = confusion_matrix(y_test, y_pred_knn)
report = classification_report(y_test, y_pred_knn)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'],
            square=True,  
            linewidths=0.5,  
            linecolor='black'  
           )
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

y_pred_prob = knn.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

mesh_size = 0.02 
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')  # Adjust alpha and use a different colormap
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', marker='o')
plt.title('KNN Decision Boundary (PCA-reduced data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

