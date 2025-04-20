#%% md
# Loading Data
#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
# Import the necessary libraries to carry out data cleaning and analysis.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
# Configure the pandas library to always show every column and record if not specified, and round figures to 2 decimal places.

path = ".."
# This term is defined to tell jupyter notebooks to step back a folder and work from there when importing data.

df = pd.read_csv(f"{path}/data/1-raw/lending-club-2007-2020Q3/Loan_status_2007-2020Q3-100ksample.csv")
# Import our dataset as a dataframe.

df.head()
# Show the first 5 records of my dataframe with all columns.
#%% md
# Data Cleaning & Feature Engineering
#%%
list_to_drop = ['Unnamed: 0.1', 'Unnamed: 0']
df = df.drop(list_to_drop, axis = 1)
#%% md
I can see from the preview of the dataframe that pymnt_plan has "n" for each entry. I used the following code to check if all entries are "n" for pymnt_plan.
#%%
df.pymnt_plan.value_counts()
#%% md
Drop pymnt_plan because all entries are "n". Create a new dataframe without the dropped feature so I don't get confused.
#%%
df = df.drop(["pymnt_plan"], axis=1)
#%% md
To double check I have successfully dropped the pymnt_plan feature.
#%%
df.head(5)
#%%
df.isnull().sum()
#%%
df.revol_util.isnull()
#%% md
I can see from the few lines of code above that revol_util returns null values for some customers. I will assume that this is due to customers not owning a credit card, allowing them access to revolving credit. In this scenario it would be best to impute these null values to 0 given that customers are technically not utilising any revolving credit.
#%%
df['revol_util'].fillna('0%', inplace=True)
#%%
print(f"Nulls after filling: {df['revol_util'].isnull().sum()}")
#%% md
I have used the 2 previous lines of code to impute the nulls, replacing them with 0, and check if this was successful.
#%%
df.revol_util.describe()
# Investigate the revolving balance utilisation feature.
#%%
df['revol_util_clean'] = df['revol_util'].str.rstrip('%').astype('float') / 100.0
df.head(15)
#%%
df.revol_util_clean.fillna(0, inplace=True)
df.head(15)
#%%
df['int_rate_clean'] = df['int_rate'].str.rstrip('%').astype('float') / 100.0
df.head(15)
#%%
df.int_rate_clean.fillna(0, inplace=True)
df.head(5)
#%% md
Because revol_util and int_rate are regarded as an object, I have stripped the '%' sign and converted it to a float like we did with the interest rate in class.
#%%
df['loan_default'] = df.loan_status == "Charged Off"
df['loan_default'].value_counts()
#%%
df.head(5)
#%%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['grade_encoded'] = le.fit_transform(df['grade'])
df[['grade', 'grade_encoded']].head()
#%%
df.head(10)
#%%
df['term_numeric'] = df['term'].str.strip(" months").astype("float")
#%%
df.head(5)
#%%
df.application_type.value_counts()
#%% md
I can see in the preview above that application types is a binary feature. Either the application is for an individual (0) or joint (1). It would be disingenuous for me to pull insights from the dataset and generalise this for joint applications which make up 7% of loans. Hence why I am going to drop all records for joint applications and focus on individual loan types. I will create a new dataframe for clarity.
#%%
df_individual = df[df['application_type'] != 'Joint App']
df_individual.application_type.value_counts()
#%% md
The above code has filtered the dataset for only individual type applications to avoid my analysis being distorted/biased by joint applications.
#%%
df_individual.describe()
#%% md
Now that I have made some changes to the dataset, I will investigate the outliers. I don't expect too much of a change compared to when Andrea completed it in class, but I will check if anything has changed drastically.
#%%
from scipy import stats

z_scores = stats.zscore(df_individual["revol_util_clean"])
sns.displot(z_scores, kind='kde')
#%% md
The graph above shows me how the standard deviations of my data within the revol_util_clean column are distributed. I can see that they are almost completely normally distributed with some outliers on the right.
#%%
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores)>thresh].shape[0]} outliers ")

# This will tell me how many outliers within my target feature there are for different thresholds of standard deviations.
#%% md
I can see that there are only 622 outliers at the 2.0 standard deviation threshold, out of 92,834 records, and even less for subsequent thresholds. This is less than 1% (~0.67%), I will be capping these outliers using the IQR method (1.5 * IQR).
#%%
q1 = df_individual['revol_util_clean'].quantile(0.25)
q3 = df_individual['revol_util_clean'].quantile(0.75)
iqr = q3 - q1
# I have calculated the 25th percentile, 75th percentile and inter-quantile range (iqr)
#%%
print(round(q1, 2), round(q3, 2), round(iqr, 2))
# I have rounded the values for q1, q3, and IQR for simplicity.
#%%
upper_limit = q3 + (1.5 * iqr)
lower_limit = q1 - (1.5 * iqr)
round(lower_limit, 2), round(upper_limit, 2)
#%%
sns.boxplot(df_individual['revol_util_clean'])
#%% md
I can see that there some outliers to the right (top) of my boxplot. I will impute these outliers to be capped at the upper limit. I will also cap outliers to the left at the lower (bottom) limit, even though there aren't any for consistency.
#%%
z_scores_1 = stats.zscore(df_individual["annual_inc"])
sns.displot(z_scores_1, kind='kde')
#%%
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores_1)>thresh].shape[0]} outliers ")
q1_1 = df_individual['annual_inc'].quantile(0.25)
q3_1 = df_individual['annual_inc'].quantile(0.75)
iqr_1 = q3_1 - q1_1
print(round(q1_1, 2), round(q3_1, 2), round(iqr_1, 2))
upper_limit_1 = q3_1 + (1.5 * iqr_1)
lower_limit_1 = q1_1 - (1.5 * iqr_1)
round(lower_limit_1, 2), round(upper_limit_1, 2)
sns.boxplot(df_individual['annual_inc'])
#%%
z_scores_2 = stats.zscore(df_individual["loan_amnt"])
sns.displot(z_scores_2, kind='kde')
#%%
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores_2)>thresh].shape[0]} outliers ")
q1_2 = df_individual['loan_amnt'].quantile(0.25)
q3_2 = df_individual['loan_amnt'].quantile(0.75)
iqr_2 = q3_2 - q1_2
print(round(q1_2, 2), round(q3_2, 2), round(iqr_2, 2))
upper_limit_2 = q3_2 + (1.5 * iqr_2)
lower_limit_2 = q1_2 - (1.5 * iqr_2)
round(lower_limit_2, 2), round(upper_limit_2, 2)
sns.boxplot(df_individual['loan_amnt'])
#%%
z_scores_3 = stats.zscore(df_individual["dti"])
sns.displot(z_scores_3, kind='kde')
#%%
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores_3)>thresh].shape[0]} outliers ")
q1_3 = df_individual['dti'].quantile(0.25)
q3_3 = df_individual['dti'].quantile(0.75)
iqr_3 = q3_3 - q1_3
print(round(q1_3, 2), round(q3_3, 2), round(iqr_3, 2))
upper_limit_3 = q3_3 + (1.5 * iqr_3)
lower_limit_3 = q1_3 - (1.5 * iqr_3)
round(lower_limit_3, 2), round(upper_limit_3, 2)
sns.boxplot(df_individual['dti'])
#%%
z_scores_4 = stats.zscore(df_individual["int_rate_clean"])
sns.displot(z_scores_4, kind='kde')
#%%
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores_4)>thresh].shape[0]} outliers ")
q1_4 = df_individual['int_rate_clean'].quantile(0.25)
q3_4 = df_individual['int_rate_clean'].quantile(0.75)
iqr_4 = q3_4 - q1_4
print(round(q1_4, 2), round(q3_4, 2), round(iqr_4, 2))
upper_limit_4 = q3_4 + (1.5 * iqr_4)
lower_limit_4 = q1_4 - (1.5 * iqr_4)
round(lower_limit_4, 2), round(upper_limit_4, 2)
sns.boxplot(df_individual['int_rate_clean'])
#%%
z_scores_5 = stats.zscore(df_individual["term_numeric"])
sns.displot(z_scores_5, kind='kde')
#%%
z_scores_5 = stats.zscore(df_individual["term_numeric"])
sns.displot(z_scores_5, kind='kde')
for threshold in range(200, 301, 25):
    thresh = threshold/100
    print(f"Threshold == {thresh}, {df_individual[np.abs(z_scores_5)>thresh].shape[0]} outliers ")
q1_5 = df_individual['term_numeric'].quantile(0.25)
q3_5 = df_individual['term_numeric'].quantile(0.75)
iqr_5 = q3_5 - q1_5
print(round(q1_5, 2), round(q3_5, 2), round(iqr_5, 2))
upper_limit_5 = q3_5 + (1.5 * iqr_5)
lower_limit_5 = q1_5 - (1.5 * iqr_5)
round(lower_limit_5, 2), round(upper_limit_5, 2)
sns.boxplot(df_individual['term_numeric'])
#%%
# I am going to denote a new dataframe for simplicity by copying the dataframe, df_dropped_individual
df_clean = df_individual.copy()
df_clean.loc[(df_clean['revol_util_clean']>upper_limit), 'revol_util_clean'] = upper_limit
df_clean.loc[(df_clean['revol_util_clean']<lower_limit), 'revol_util_clean'] = lower_limit
# I have capped outliers to the lower and upper limit defined previously
#%%
sns.boxplot(df_clean['revol_util_clean'])
#%% md
I can see the outliers have been successfully capped at 1.5 times the IQR.
#%%
df_clean.loc[(df_clean['annual_inc']>upper_limit_1), 'annual_inc'] = upper_limit_1
df_clean.loc[(df_clean['annual_inc']<lower_limit_1), 'annual_inc'] = lower_limit_1
sns.boxplot(df_clean['annual_inc'])
#%%
df_clean.loc[(df_clean['loan_amnt']>upper_limit_2), 'loan_amnt'] = upper_limit_2
df_clean.loc[(df_clean['loan_amnt']<lower_limit_2), 'loan_amnt'] = lower_limit_2
sns.boxplot(df_clean['loan_amnt'])
#%%
df_clean.loc[(df_clean['dti']>upper_limit_3), 'dti'] = upper_limit_3
df_clean.loc[(df_clean['dti']<lower_limit_3), 'dti'] = lower_limit_3
sns.boxplot(df_clean['dti'])
#%%
df_clean.loc[(df_clean['int_rate_clean']>upper_limit_4), 'int_rate_clean'] = upper_limit_4
df_clean.loc[(df_clean['int_rate_clean']<lower_limit_4), 'int_rate_clean'] = lower_limit_4
sns.boxplot(df_clean['int_rate_clean'])
#%%
df_clean.loc[(df_clean['term_numeric']>upper_limit_5), 'term_numeric'] = upper_limit_5
df_clean.loc[(df_clean['term_numeric']<lower_limit_5), 'term_numeric'] = lower_limit_5
sns.boxplot(df_clean['term_numeric'])
#%% md
Now I am going to create a new feature. This feature will grade the customers' utilisation rates, based on general intervals. I will assume SME's provided the following intervals underscored by a consensus in the banking sector:
- Utilisation rate =< 30% = Low Utilisation;
- 30% < Utilisation rate =< 60% = Moderate Utilisation;
- 60% < Utilisation rate =< 90% = High Utilisation;
- Utilisation rate > 90% = Very High Utilisation.
#%%
from pandas import Series, DataFrame

df_clean['revol_util_cat'] = pd.cut(df_clean['revol_util_clean'],
      bins=[-1, 0.3, 0.6, 0.9, 2],
      labels=['Low Utilisation', 'Moderate Utilisation', 'High Utilisation', 'Very High Utilisation'])
#%%
df_clean.loc[:,['revol_util_clean', 'revol_util_cat']].head(100)
#%%
df_clean['revol_util_cat']
#%%
df_clean.revol_util_cat.value_counts()
#%% md
Using the few lines of code above I have categorised the revolving credit utilisation rates into the categories mentioned previously. After reading the underlying settings of the pd.cut() operator, I know it assumes the right boundary as inclusive and infers from the data that my categories are ordered.
#%% md
Now I am going to investigate which types of loans customers with 'High' and 'Very High' utilisation rates take out. This will help me understand what loans are opening the bank up to increased exposure compared to others.
#%%
df_clean.head(50)
#%%
df_clean.purpose.value_counts()
#%% md
# Visualisation

Create a stacked bar plot of loan purposes for high and very high credit utilisation.
#%%
# Filter for only High and Very High utilization
high_risk_customs = df_clean[df_clean['revol_util_cat'].isin(['Low Utilisation', 'Moderate Utilisation', 'High Utilisation', 'Very High Utilisation'])]

# Create pivot table for plotting
plot_data = pd.pivot_table(
    high_risk_customs,
    index='purpose',
    columns='revol_util_cat',
    aggfunc='size',
    fill_value=0
)

# Create figure and axis with larger size
plt.figure(figsize=(100, 50))

# Create stacked bar plot
plot_data.plot(
    kind='bar',
    stacked=True,
    color=['#FF00FF', '#00FF00', '#0000FF', '#FF4500'],
    width=0.8
)

# Customize the plot
plt.title('Loan Purposes by Credit Utilisation (High & Very High)', pad=20)
plt.xlabel('Loan Purpose')
plt.ylabel('Number of Customers')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add legend with better positioning
plt.legend(title='Credit Utilisation', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
#%%
results = df_clean.loc[(df_clean.loan_status == 'Charged Off')]
results.head(10)
#%% md
I located all the default records using the df.loc code. I did this to try and find any patterns based on face value. This would speed up the process of picking features rather than going through numerous iterations of which combination of features showcase better predictive power.
#%%
df_clean.home_ownership.value_counts()
#%%
df_clean['home_ownership_numeric'] = df_clean['home_ownership']
df_clean['home_ownership_numeric'].replace(['OTHER', 'ANY', 'NONE', 'OWN', 'RENT', 'MORTGAGE'],
                        [1, 1, 1, 2, 3, 4], inplace=True)
df_clean.head(10)
#%%
df_clean.loc[(df_clean['home_ownership'] == 'OTHER') | (df_clean['home_ownership'] == 'ANY') | (df_clean['home_ownership'] == 'NONE')]
#%%
df_clean['home_ownership_numeric']
# to check the code hasn't interpreted the transformation as ordinal.
#%%
df_clean.home_ownership_numeric.value_counts()
#%% md
As you can see from the lines of code above, I decided that home_ownership looked quite interesting. Hence, why I created a numeric version of the column. For simplicity I included 'OTHER', 'ANY', and 'NONE', into the same category (1). The last line of code was to double check no records had been lost in the process.
#%% md
# Summary Statistics

Now that the outliers within our control features have been dealt with, I will check the summary stats.
#%%
df_clean.describe()
#%%
df_clean.to_csv(f"{path}/data/2-intermediate/df_out_dsif_capstone_project.csv"
                        , index = False)
#%% md
# Baseline Model
#%% md
# Model Development & Feature Selection
#%%
features = ['fico_range_high', 'fico_range_low', 'annual_inc', 'dti',
            'int_rate_clean', 'term_numeric', 'revol_util_clean',
            'loan_amnt', 'grade_encoded', 'home_ownership_numeric']

X = df_clean[features]
y = df_clean['loan_default']
#%% md
I have chosen 10 features to include in my baseline model. Below is my reasoning for each:

fico_range_high - This feature represents the applicants highest credit score. Applicants with higher credit scores are assumed to be safer bets to lend to. This feature could have some beneficial predictive power.

fico_range_low - This feature represents the applicants lowest credit score. Applicants with lower credit scores are assumed to be riskier bets to lend to. 

annual_inc - A person's annual income is a good indicator of whether the individual will be able to afford the payments due on the loan or not. 

dti - Debt to income ratio gives us a good understanding of how well this individual manages their finances. Individuals with higher debt to income ratios can be deemed as riskier to lend to given their existing commitments to other loans.

int_rate_clean - This feature always provides valuable info. with regards to defaults. Risk premiums are usually included in what's charged as interest for riskier loans.

term_numeric - This feature shows the term of each loan. We might be able to interpret defaulters using loan terms. It could be argued that shorter term loans have a higher risk of default because of its purpose, the individuals who take out short term loans etc.

revol_util_clean - I brought this feature over from my first assignment. It is common knowledge that customers with high credit utilisation are usually worse with money management. Therefore, this might be a good indicator for loan defaults.

loan_amnt - This is pretty simple. The greater the loan amount, usually the greater the burden on a customer's finances. This feature would provide us interesting information, particularly when we are controlling for debt-to-income ratios also.

grade_encoded - This is a numeric version of the grade feature. Grade tells us which loans are riskier and which are safer. Always has been a good indicator of defaults.

home_ownership_numeric - This is a numeric version of the home_ownership feature. I chose this feature as well purely because of an oversimplification and assumption; a customer who own's their home is less risky because they are not committed to a monthly outgoing for housing, a customer who rent's is riskier than an owner because they have significant monthly outgoings, but they can move and change this, and a customer with a mortgage is riskiest because they are committed to usually paying a significant amount of their earnings for the mortgage. 'ANY', 'OTHER', and 'NONE' were grouped together because the dictionary provided no definition for them. They were small in count so it shouldn't affect the model significantly.
#%%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
for dframe in [X_train, X_test, y_train, y_test]:
    print(f"Shape: {dframe.shape}")
#%%
model = LogisticRegression()
#%%
model.fit(X_train, y_train)
#%%
predictions = model.predict(X_test)
predictions
#%% md
The few lines of code above have utilised logistic regression modelling in this instance. I then produced the predictions, and as you can see the model isn't all that good. It looks like all the predictions are False, just like we saw in class. I am going to produce some evaluation metrics below to see what we're actually working with.
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'ROC-AUC: {roc_auc}')
print(f'Confusion Matrix:\n{cm}')
#%%
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random prediction)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.2f}')
#%% md
Above is just a simple visualisation of the ROC Curve.
#%% md
# Resampling (Handling Class Imbalance)
#%%
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
#%%
y_resampled.head(100)
#%%
y_resampled.value_counts()
#%%
X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
#%%
for dframe in [X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test]:
    print(f"Shape: {dframe.shape}")
#%% md
Above, I have resampled my data by producing synthetic data. As you can see the no. of records has increased to ~162k. I can see that the resampling method has significantly increased the respresentation of default records.
#%%
model.fit(X_resampled_train, y_resampled_train)
#%%
predictions_resampled = model.predict(X_resampled_test)
predictions_resampled
#%% md
I have produced predictions for my resampled data, and I can see that it is actually predicting some correct values.
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy_resampled = accuracy_score(y_resampled_test, predictions_resampled)
precision_resampled = precision_score(y_resampled_test, predictions_resampled)
recall_resampled = recall_score(y_resampled_test, predictions_resampled)
f1_resampled = f1_score(y_resampled_test, predictions_resampled)
roc_auc_resampled = roc_auc_score(y_resampled_test, model.predict_proba(X_resampled_test)[:,1])

cm = confusion_matrix(y_resampled_test, predictions_resampled)

print(f'Accuracy: {accuracy_resampled}')
print(f'Precision: {precision_resampled}')
print(f'Recall: {recall_resampled}')
print(f'F1-Score: {f1_resampled}')
print(f'ROC-AUC: {roc_auc_resampled}')
print(f'Confusion Matrix:\n{cm}')
#%% md
After handling the class imbalance between non-defaulters and defaulters by producing synthetic data and increasing the population size, my model has decreased in accuracy significantly. I know accuracy is calculated by (TP + TN) / Total predictions. With a large increase in False Positives, it makes sense for the model with the new synthetic data to be less accurate. However, the precision score has increased by ~23 percentage points. This is very good as now the model is better at correctly predicting Positives.
#%%
y_resampled_pred_prob = model.predict_proba(X_resampled_test)[:, 1]
fpr_resampled, tpr_resampled, thresholds_resampled = roc_curve(y_resampled_test, y_resampled_pred_prob)
roc_auc_resampled = auc(fpr_resampled, tpr_resampled)
plt.figure()
plt.plot(fpr_resampled, tpr_resampled, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random prediction)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Resampled)')
plt.grid()
plt.show()
print(f'ROC AUC Score: {roc_auc_score(y_resampled_test, y_resampled_pred_prob):.2f}')
#%% md
Another visualisation for the ROC Curve, this time for the synthetic data for handling class imbalance. As you can see the ROC-AUC score has increased to 71%, whereas my initial ROC-AUC score was 65%.
#%% md
# Cross Validation
#%%
from sklearn.model_selection import cross_val_score

# Perform 5-Fold Cross-Validation
cv_scores_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_scores_prec = cross_val_score(model, X, y, cv=5, scoring='precision')
cv_scores_rec = cross_val_score(model, X, y, cv=5, scoring='recall')
cv_scores_roc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

print(f'Cross-Validation Accuracy Scores: {cv_scores_acc}')
print(f'Mean CV Accuracy: {cv_scores_acc.mean()}')
print(f'Cross-Validation Precision Scores: {cv_scores_prec}')
print(f'Mean CV Precision: {cv_scores_prec.mean()}')
print(f'Cross-Validation Recall Scores: {cv_scores_rec}')
print(f'Mean CV Recall: {cv_scores_rec.mean()}')
print(f'Cross-Validation ROC-AUC Scores: {cv_scores_roc}')
print(f'Mean CV ROC-AUC: {cv_scores_roc.mean()}')
#%% md
Above, I produced the cross validation scores of my initial model (without resampling) with a 5 fold cross-validation. This will simply take 5 different splits of training and testing sets from the dataframe and produce the evaluation metrics I want (Accuracy, Precision, Recal, ROC-AUC Score). It will then calculate the mean for each score.

After comparing the cross validation scores, I can see that my model performs better across different training and testing samples, with greater precision, recall and ROC scores.
#%% md
# Challenger Model
#%% md
## Data Preprocessing

Below I have removed all string type features as well as a few integer type features. This is because they either did not provide any valuable information about customers defaulting, or they were created by Andrea in class and I would rather use my own additional features. The type of modelling I am about to complete doesn't work very well when nulls are present in the dataset. This is because it can lead to innacurate training of the model. Hence I am going to be imputing these nulls using the algorithm shown in class.
#%%
df_challenger = df_clean.select_dtypes(exclude=['object'])\
                .drop(columns=["out_prncp", "out_prncp_inv",
                               "total_pymnt", "total_pymnt_inv", "funded_amnt", "funded_amnt_inv"])

print(f"Number of columns: {len(df_challenger.columns)}")
#%%
pip install keras
#%%
pip install tensorflow
#%%
df_challenger.shape
#%% md
The first part of the algorithm replaces all nulls in the specified columns with 0. The second part replaces all nulls in the specified columns with the features median value. The third part replaces the null values within the feature, 'mths_since_rcnt_il', with 999 because the customer may never have opened an account, hence we replace the null with such a large number assuming that the model interprets this extremely large number as a proxy for 'never'. I have then produced a list of all features which have nulls present.
#%%
# Group 1: Columns to be filled with 0
# Assuming these features represent counts or amounts where a missing value means zero
fill_zero_cols = [
    'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
    'open_act_il', 'open_il_12m', 'open_il_24m',
    'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
    'all_util', 'inq_fi', 'total_cu_tl', 'acc_open_past_24mths',
    'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
    'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
    'inq_last_12m', 'pub_rec_bankruptcies', 'mths_since_last_delinq', 'mths_since_recent_revol_delinq'
]
df_challenger[fill_zero_cols] = df_challenger[fill_zero_cols].fillna(0)

# Group 2: Columns to be filled with median
# (Assuming these features are numeric and can have a central tendency)
fill_median_cols = [
    'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim',
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_rev_hi_lim'
]
df_challenger[fill_median_cols] = df_challenger[fill_median_cols].apply(lambda x: x.fillna(x.median()))

# Group 3: Fill with 999
fill_nine_cols = [
     'mths_since_rcnt_il'
]
# Apply fill with 999
df_challenger[fill_nine_cols] = df_challenger[fill_nine_cols].fillna(999)

# Do a quick check on null values
df_challenger.columns[df_challenger.isna().sum() > 0]
#%% md
After replacing the null values in most of the columns I had a look at what other columns had a significantly large number of null values.
#%%
df_challenger.isnull().sum()
#%%
print(f"Number of columns: {len(df_challenger.columns)}")
#%% md
Previously, I had checked for features with significantly large null counts. I noticed that most of the features relating to joint applications showed null values above 75%. Hence, why I chose to set my threshold to 75%. The next piece of code removes all features will null values making up 75% of their data or more.

Caveat: I did not go and manually check every single feature and whether it would be more appropriate to replace nulls with 0, the median, 999, or drop the feature. I believe this would be extremely onerous for some 30+ features. I have completed a few steps to show that I understand the concept of this method with regards to preporcessing my data, and the fact that it is necessary to allow the model to train accurately.
#%%
threshold = 0.75
null_percentages = df_challenger.isnull().mean()
df_challenger = df_challenger.drop(columns=null_percentages[null_percentages > threshold].index)
print(f"Dropped columns: {null_percentages[null_percentages > threshold].index.tolist()}")
print(f"Number of columns: {len(df_challenger.columns)}")
#%%
df_challenger.isnull().sum()
# to check my data preprocessing has worked
#%%
df_challenger.loan_default.value_counts()
#%%
df_challenger.columns.tolist()
# to see what columns might be interesting enough to include in my model
#%%
features = ['fico_range_high', 'fico_range_low', 'annual_inc',
            'dti', 'int_rate_clean', 'term_numeric', 'revol_util_clean',
            'loan_amnt', 'grade_encoded', 'home_ownership_numeric',
            'delinq_2yrs', 'open_acc', 'recoveries', 'tot_cur_bal',
            'mort_acc', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
            'delinq_amnt', 'last_fico_range_high', 'last_fico_range_low']

X = df_challenger[features]
y = df_challenger['loan_default']

print(f"Number of features: {len(features)}")
#%%
X_resampled_1, y_resampled_1 = smote.fit_resample(X, y)
#%%
X.head()
#%%
from sklearn.preprocessing import MinMaxScaler

# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled_1)

# Convert back to DataFrame with same column names
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df.head()
#%%
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (returns pandas dfs)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_scaled_df, y_resampled_1, test_size=0.2, random_state=23)
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialize the model
model_3 = Sequential()

# First hidden layer
model_3.add(Dense(16, input_dim=X_scaled_df.shape[1], activation='relu'))

# Second hidden layer
model_3.add(Dense(10, activation='relu'))

# Third hidden layer
model_3.add(Dense(6, activation='relu'))

# Output layer
model_3.add(Dense(1, activation='sigmoid'))

# Compile the model
model_3.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

history_3 = model_3.fit(X_train_1, y_train_1, epochs=10, batch_size=32, validation_split=0.2)
#%%
y_prob3 = model_3.predict(X_test_1)
y_pred3 = (y_prob3 > 0.5)
from dsif6utility import model_evaluation_report
model_evaluation_report(X_test_1, y_test_1, y_pred3, y_prob3)
#%%
y_prob3 = pd.DataFrame(y_prob3, columns=["prob3"])
y_prob3.prob3.describe(percentiles = [i / 100 for i in [0, 1, 10, 25, 50, 75, 90, 95, 99, 100]])
#%%
history_3.history['val_AUC']
#%%
def plot_training_vs_overfitting(history_3):
    """Plot training and validation accuracy to detect overfitting (when gap between 2 is detected)"""
    import matplotlib.pyplot as plt

    # Plot accuracy
    plt.plot(history_3.history['accuracy'], label='Train Accuracy')
    plt.plot(history_3.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

     # Plot precision
    plt.plot(history_3.history['Precision'], label='Train Precision')
    plt.plot(history_3.history['val_Precision'], label='Validation Precision')
    plt.legend()
    plt.show()

    # Plot recall
    plt.plot(history_3.history['Recall'], label='Train Recall')
    plt.plot(history_3.history['val_Recall'], label='Validation Recall')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history_3.history['loss'], label='Train Loss')
    plt.plot(history_3.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

plot_training_vs_overfitting(history_3)
