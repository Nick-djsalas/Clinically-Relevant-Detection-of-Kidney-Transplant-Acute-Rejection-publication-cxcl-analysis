from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import os
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns
from scipy import interp
import statsmodels.api as sm

# Part 1: Calculate the ROC curves and the Youden Indices, and save the data
# Part 2: Calculate the Feature Importance based on AUC-ROC scores, and save the data
# Part 3: Calculate the Pearson Correlation Coefficient Matrix and plot it as a heatmap, and save the data
# Part 4: Calculate the p-values and ORs (Odd Ratios) for all the features with a logistic regression

########################  PART 1: Calculate the ROC curves and the Youden Indices ########################

# Define the mapping dictionary globally with unique mapped names
feature_mapping = {
    'BIOPSY RESULT - ACUTE REJECTION [Y=1, N=0]': 'Acute Rejection',
    'RECIPIENT AGE (YEARS)': 'Recipient Age',
    'Donor Age (Years)': 'Donor Age',
    'RECIPIENT SEX [1=M, 2=F]_2.0': 'Sex',
    'Transplant to biopsy interval (days)': 'Time to biopsy',
    'Serum Cr at time of biopsy [μmol/L]': 'sCr',
    'Urinary PCR at time of biopsy [g/g]': 'uPCR',
    'Urinary Creatinine at time of biopsy [mmol/L]': 'uCR',
    # 'DSA  [YES=1, NO=2]_1.0': 'DSA: Yes',
    'DSA  [YES=1, NO=2]_2.0': 'DSA',
    'cRF %': 'cRF %',
    'Total Number of HLA mismatches': 'HLA mismatches',
    'HLA-A [mm]': 'HLA-A',
    'HLA-B [mm]': 'HLA-B',
    'HLA-DR  [mm]': 'HLA-DR',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_1.0': 'Ethnicity: White',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_2.0': 'Ethnicity: Asian',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_3.0': 'Ethnicity: Afro-Caribbean',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_4.0': 'Ethnicity: Other',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]_1.0': 'Donor Type: Live',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]_2.0': 'Donor Type: DBD',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]_3.0': 'Donor Type: DCD',
    # Add additional mappings as necessary
}

# Updated feature_name_mapping function
def feature_name_mapping(feature_name):
    return feature_mapping.get(feature_name, feature_name)

# Create a reverse mapping dictionary
reverse_mapping = {v: k for k, v in feature_mapping.items()}

# Function to calculate Youden's index and plot ROC curve using sklearn
def calculate_youden_and_plot(X_features, y_target, save_path, title):
    # Use logistic regression as the model
    model = LogisticRegression(max_iter=100, solver='liblinear')

    # Fit the model to the features
    model.fit(X_features, y_target)

    # Use the model to calculate the probabilities
    y_probabilities = model.predict_proba(X_features)[:, 1]

    # Use sklearn's roc_auc_score to compute Area Under the Curve for ROC curves
    auc_score = roc_auc_score(y_target, y_probabilities)

    # Use sklearn's roc_curve to compute TPR (True Positive Rate), FPR (False Positive Rate), and Probability Thresholds from the model and the target
    fpr, tpr, thresholds = roc_curve(y_target, y_probabilities)

    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate predictions based on the optimal threshold
    y_pred = (y_probabilities >= optimal_threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_target, y_pred).ravel()

    # Calculate Sensitivity, Specificity, PPV, NPV
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0

    ### Plot the figure
    plt.figure(figsize=(12, 9))

    # Diagonal line which indicates a random model
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3)

    plt.plot(fpr, tpr, label=f'{title} (AUC = {auc_score:.3f})', color='blue', linewidth=4.5)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=200,
                label=f'Youden Index \n(FPR={fpr[optimal_idx]:.2f}, TPR={tpr[optimal_idx]:.2f})')

    # Enhancing the plot for thicker lines and bold labels
    plt.xlabel('False Positive Rate', fontsize=32, fontweight='bold', labelpad=15)
    plt.ylabel('True Positive Rate', fontsize=32, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=26, fontweight='bold')

    # Add legend
    legend = plt.legend(loc='lower right', fontsize=21.5, frameon=False)

    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Increase linewidth of the outer plot box
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # Adjust the linewidth as needed

    # SAVE THE PLOT
    plt.savefig(save_path)
    plt.close()
    ###

    # Save ROC data for further analysis
    roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})

    # Return relevant metrics
    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx], roc_data, sensitivity, specificity, ppv, npv, y_probabilities, tn, fp, fn, tp

# Alternatively Choose the file from Windows File Explorer prompt
# Ask user to select the Excel file
Tk().withdraw()  # Hide the root window
excel_file_path = askopenfilename(title="Select the Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])

# Read the data from the excel file path
data = pd.read_excel(excel_file_path)

# Define required features
features = [
    'ELISA: CXCL10',
    'ELISA: CXCL9',
    'EB: CXCL10',
    'EB: CXCL9',
    'RECIPIENT AGE (YEARS)',
    'RECIPIENT SEX [1=M, 2=F]',
    'DSA  [YES=1, NO=2]',
    'Transplant to biopsy interval (days)',
    'Serum Cr at time of biopsy [μmol/L]',
    'Urinary PCR at time of biopsy [g/g]',
    'Urinary Creatinine at time of biopsy [mmol/L]',
    # 'Bacteruiria (>10,000cfu/ml) near time of biospy [YES=1, NO=2]',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]',
    'Donor Age',
    'cRF %',
    'HLA-A [mm]',
    'HLA-B [mm]',
    'HLA-DR  [mm]',
    'Total Number of HLA mismatches',
]

#define the target of the model
target_column = 'BIOPSY RESULT - ACUTE REJECTION [Y=1, N=0]'

#define categorical features
categorical_features = [
    'RECIPIENT SEX [1=M, 2=F]',
    'DSA  [YES=1, NO=2]',
    # 'Bacteruiria (>10,000cfu/ml) near time of biospy [YES=1, NO=2]',
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]',
]

#define numerical features
numerical_features = [
    'ELISA: CXCL10',
    'ELISA: CXCL9',
    'EB: CXCL10',
    'EB: CXCL9',
    'RECIPIENT AGE (YEARS)',
    'Donor Age (Years)',
    'Transplant to biopsy interval (days)',
    'Serum Cr at time of biopsy [μmol/L]',
    'Urinary PCR at time of biopsy [g/g]',
    'Urinary Creatinine at time of biopsy [mmol/L]',
    'cRF %',
    'HLA-A [mm]',
    'HLA-B [mm]',
    'HLA-DR  [mm]',
    'Total Number of HLA mismatches'
]

# Handle missing values

# Convert numeric columns to numeric values (force conversion of date-like strings)
for col in numerical_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Impute numerical features with the mean value
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_features] = numerical_imputer.fit_transform(data[numerical_features])

# Impute missing values in categorical columns with the most frequent category
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

# Clean and ensure categorical data are consistent for encoding
for col in categorical_features:
    data[col] = data[col].astype(str).str.strip()  # Convert to string and strip spaces

# Encode categorical features after imputation
categorical_features_encoded = pd.get_dummies(data[categorical_features], drop_first=True)

# Combine numerical features and encoded categorical features
data_encoded = pd.concat([data[numerical_features], categorical_features_encoded], axis=1)

# Define features for full model with ELISA
X_full_ELISA = data_encoded.drop(columns=['EB: CXCL10', 'EB: CXCL9'])

# Define features for full model with EB
X_full_EB = data_encoded.drop(columns=['ELISA: CXCL10', 'ELISA: CXCL9'])

# Define features for ELISA measurements only
X_ELISA = data_encoded[['ELISA: CXCL10', 'ELISA: CXCL9']]

# Define features for EB measurements only
X_EB = data_encoded[['EB: CXCL10', 'EB: CXCL9']]

# Define desktop path to save the data
desktop_path = "C:/Users/homeuser/Desktop/Paper_data/"

# Ensure the desktop_path exists
os.makedirs(desktop_path, exist_ok=True)

# Compute the ROC curves and the Youden indices and Plot the Models

# 1. Full Model using ELISA CXCL10 and CXCL9 measurements
FULL_MODEL_ELISA = calculate_youden_and_plot(
    X_full_ELISA, data[target_column], desktop_path + 'ROC_Curve_FULL_MODEL_ELISA.png',
    title="Full Model ELISA"
)

# 2. Full Model with EB CXCL10 and CXCL9 measurements
FULL_MODEL_EB = calculate_youden_and_plot(
    X_full_EB, data[target_column], desktop_path + 'ROC_Curve_FULL_MODEL_EB.png',
    title="Full Model EB"
)

# 3. Model using only the features of ELISA: CXCL10 and CXCL9
ELISA_ONLY = calculate_youden_and_plot(
    X_ELISA, data[target_column], desktop_path + 'ROC_Curve_ELISA_ONLY.png',
    title="ELISA: CXCL9 and CXCL10"
)

# 4. Model using only the features of EB: CXCL10 and CXCL9
EB_ONLY = calculate_youden_and_plot(
    X_EB, data[target_column], desktop_path + 'ROC_Curve_EB_ONLY.png',
    title="EB: CXCL9 and CXCL10"
)

##### Plot the ROC curves in the main part of the paper - ELISA and EB in the same plots for the full models and when only using CXCL9 and CXCL10

## Plot ROC curves for ELISA and EB measurements in the same plot (only CXCL9 and CXCL10)
plt.figure(figsize=(12.8, 10.5))
# Diagonal line which indicates a random model
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3)
# Plot ELISA only
fpr_elisa, tpr_elisa, _ = roc_curve(data[target_column], ELISA_ONLY[8])
youden_index_elisa = tpr_elisa - fpr_elisa
optimal_idx_elisa = np.argmax(youden_index_elisa)
plt.plot(fpr_elisa, tpr_elisa, label='ELISA (AUC = {:.3f})'.format(roc_auc_score(data[target_column], ELISA_ONLY[8])), color='blue', linewidth=7)
plt.scatter(fpr_elisa[optimal_idx_elisa], tpr_elisa[optimal_idx_elisa], color='blue', s=200,
            label=f'ELISA Youden Index \n(FPR={fpr_elisa[optimal_idx_elisa]:.2f}, TPR={tpr_elisa[optimal_idx_elisa]:.2f})')
# Plot EB only
fpr_eb, tpr_eb, _ = roc_curve(data[target_column], EB_ONLY[8])
youden_index_eb = tpr_eb - fpr_eb
optimal_idx_eb = np.argmax(youden_index_eb)
plt.plot(fpr_eb, tpr_eb, label='EB (AUC = {:.3f})'.format(roc_auc_score(data[target_column], EB_ONLY[8])), color='green', linewidth=7)
plt.scatter(fpr_eb[optimal_idx_eb], tpr_eb[optimal_idx_eb], color='green', s=200,
            label=f'EB Youden Index \n(FPR={fpr_eb[optimal_idx_eb]:.2f}, TPR={tpr_eb[optimal_idx_eb]:.2f})')
# Enhancing the plot for thicker lines and bold labels
plt.xlabel('False Positive Rate', fontsize=40, fontweight='bold', labelpad=15)
plt.ylabel('True Positive Rate', fontsize=40, fontweight='bold', labelpad=15)
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
# Add legend
legend = plt.legend(loc='lower right', fontsize=25, frameon=False)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')
# Increase linewidth of the outer plot box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(6)  # Adjust the linewidth as needed
# SAVE THE PLOT
plt.savefig(desktop_path + 'ROC_Curve_ELISA_vs_EB_ONLY.png')
plt.close()


### Plot ROC curves for Full Model ELISA and Full Model EB in the same plot
plt.figure(figsize=(12.8, 10.5))
# Diagonal line which indicates a random model
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3)
# Plot Full Model ELISA
fpr_elisa, tpr_elisa, _ = roc_curve(data[target_column], FULL_MODEL_ELISA[8])
youden_index_elisa_FULL = tpr_elisa - fpr_elisa
optimal_idx_elisa = np.argmax(youden_index_elisa_FULL)
plt.plot(fpr_elisa, tpr_elisa, label='ELISA (AUC = {:.3f})'.format(roc_auc_score(data[target_column], FULL_MODEL_ELISA[8])), color='blue', linewidth=7)
plt.scatter(fpr_elisa[optimal_idx_elisa], tpr_elisa[optimal_idx_elisa], color='blue', s=200,
            label=f'ELISA Youden Index \n(FPR={fpr_elisa[optimal_idx_elisa]:.2f}, TPR={tpr_elisa[optimal_idx_elisa]:.2f})')
# Plot Full Model EB
fpr_eb, tpr_eb, _ = roc_curve(data[target_column], FULL_MODEL_EB[8])
youden_index_eb_FULL = tpr_eb - fpr_eb
optimal_idx_eb = np.argmax(youden_index_eb_FULL)
plt.plot(fpr_eb, tpr_eb, label='EB (AUC = {:.3f})'.format(roc_auc_score(data[target_column], FULL_MODEL_EB[8])), color='green', linewidth=7)
plt.scatter(fpr_eb[optimal_idx_eb], tpr_eb[optimal_idx_eb], color='green', s=200,
            label=f'EB Youden Index \n(FPR={fpr_eb[optimal_idx_eb]:.2f}, TPR={tpr_eb[optimal_idx_eb]:.2f})')
# Enhancing the plot for thicker lines and bold labels
plt.xlabel('False Positive Rate', fontsize=40, fontweight='bold', labelpad=15)
plt.ylabel('True Positive Rate', fontsize=40, fontweight='bold', labelpad=15)
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
# Add legend
legend = plt.legend(loc='lower right', fontsize=25, frameon=False)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')
# Increase linewidth of the outer plot box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(6)  # Adjust the linewidth as needed
# SAVE THE PLOT
plt.savefig(desktop_path + 'ROC_Curve_FULL_MODEL_ELISA_vs_EB.png')
plt.close()


######### Save the data to Excel Files #########

# Save the Youden indices for all models
youden_results = pd.DataFrame({
    'Model': ['Full Model ELISA', 'Full Model EB', 'ELISA: CXCL10 and CXCL9', 'EB: CXCL10 and CXCL9'],
    'Optimal Threshold': [FULL_MODEL_ELISA[0], FULL_MODEL_EB[0], ELISA_ONLY[0], EB_ONLY[0]],
    'TPR (Sensitivity)': [FULL_MODEL_ELISA[1], FULL_MODEL_EB[1], ELISA_ONLY[1], EB_ONLY[1]],
    'FPR (1-Specificity)': [FULL_MODEL_ELISA[2], FULL_MODEL_EB[2], ELISA_ONLY[2], EB_ONLY[2]]
})

# Save Youden results to Excel
youden_results.to_excel(os.path.join(desktop_path, 'Youden_Indices.xlsx'), index=False)

## Save the ROC curves and Thresholds in Excel files
# Temporarily Write the results for each model in lists
roc_data_list = []
roc_data_list.append(('Full Model ELISA', FULL_MODEL_ELISA[3]))
roc_data_list.append(('Full Model EB', FULL_MODEL_EB[3]))
roc_data_list.append(('ELISA CXCL10 and CXCL9', ELISA_ONLY[3]))
roc_data_list.append(('EB CXCL10 and CXCL9', EB_ONLY[3]))

# Save ROC data for all models to Excel with separate sheets
with pd.ExcelWriter(os.path.join(desktop_path, 'ROC_Curves.xlsx')) as writer:
    for model_name, roc_data in roc_data_list:
        # Truncate sheet names to 31 characters if necessary
        sheet_name = model_name if len(model_name) <= 31 else model_name[:31]
        roc_data.to_excel(writer, sheet_name=sheet_name, index=False)

# Extract sensitivity, specificity, PPV, and NPV for both models
sensitivity_elisa, specificity_elisa, ppv_elisa, npv_elisa = FULL_MODEL_ELISA[4], FULL_MODEL_ELISA[5], FULL_MODEL_ELISA[6], FULL_MODEL_ELISA[7]
sensitivity_eb, specificity_eb, ppv_eb, npv_eb = FULL_MODEL_EB[4], FULL_MODEL_EB[5], FULL_MODEL_EB[6], FULL_MODEL_EB[7]

# Save the metrics for both models in an Excel file
metrics_data = {
    'Metric': ['Sensitivity', 'Specificity', 'Positive Predictive Value (PPV)', 'Negative Predictive Value (NPV)'],
    'Full Model ELISA': [sensitivity_elisa, specificity_elisa, ppv_elisa, npv_elisa],
    'Full Model EB': [sensitivity_eb, specificity_eb, ppv_eb, npv_eb]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel(desktop_path + 'Full_Model_Metrics.xlsx', index=False)

###############################################
# SAVE CONFUSION MATRIX TO EXCEL
###############################################

# The confusion matrix values are returned as indices:
# For each model, we have: ..., tn, fp, fn, tp
tn_elisa, fp_elisa, fn_elisa, tp_elisa = FULL_MODEL_ELISA[9], FULL_MODEL_ELISA[10], FULL_MODEL_ELISA[11], FULL_MODEL_ELISA[12]
tn_eb, fp_eb, fn_eb, tp_eb = FULL_MODEL_EB[9], FULL_MODEL_EB[10], FULL_MODEL_EB[11], FULL_MODEL_EB[12]
tn_elisa_only, fp_elisa_only, fn_elisa_only, tp_elisa_only = ELISA_ONLY[9], ELISA_ONLY[10], ELISA_ONLY[11], ELISA_ONLY[12]
tn_eb_only, fp_eb_only, fn_eb_only, tp_eb_only = EB_ONLY[9], EB_ONLY[10], EB_ONLY[11], EB_ONLY[12]

confusion_matrix_data = {
    'Model': ['Full Model ELISA', 'Full Model EB', 'ELISA: CXCL10 and CXCL9', 'EB: CXCL10 and CXCL9'],
    'TN': [tn_elisa, tn_eb, tn_elisa_only, tn_eb_only],
    'FP': [fp_elisa, fp_eb, fp_elisa_only, fp_eb_only],
    'FN': [fn_elisa, fn_eb, fn_elisa_only, fn_eb_only],
    'TP': [tp_elisa, tp_eb, tp_elisa_only, tp_eb_only]
}

cm_df = pd.DataFrame(confusion_matrix_data)
cm_df.to_excel(desktop_path + 'Confusion_Matrices.xlsx', index=False)






########################  PART 2: Calculate the Feature Importance based on AUC-ROC scores ########################





# Store AUC scores and standard deviations for each feature
auc_scores = []
auc_errors = []

# Define repeated cross-validation (using 4 splits for training and testing and repeating 100 times)
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=100, random_state=42)

X = data_encoded
y_target = data[target_column]

# Iterate through each feature to train a classifier individually and calculate its AUC-ROC score
for feature in X.columns:
    X_feature = X[[feature]]
    auc_list = []

    # Perform repeated cross-validation to calculate AUC and its standard deviation
    for train_index, test_index in cv.split(X_feature, y_target):
        X_train, X_test = X_feature.iloc[train_index], X_feature.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

        # Train a simple logistic regression classifier on this single feature
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        # Predict probabilities and calculate AUC-ROC score for this feature
        y_probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_probs)
        auc_list.append(auc)

    # Calculate mean AUC and standard deviation for the feature across all splits and repetitions
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    # Append results to lists
    auc_scores.append((feature, mean_auc))
    auc_errors.append(std_auc)

# Create a DataFrame to store feature names and their corresponding AUC scores and errors
auc_importance_df = pd.DataFrame(auc_scores, columns=['Feature', 'AUC-ROC']).sort_values(by='AUC-ROC', ascending=False)
auc_importance_df['Error'] = auc_errors

# Map some feature names to custom labels for plotting purposes (to have the choice of X tick labels)
# Apply mapping to feature names
auc_importance_df['Feature'] = auc_importance_df['Feature'].map(feature_name_mapping)

# Note 1: Categorical features are named after one-hot encoding as e.g. 'DSA: Yes', 'DSA: No' instead of 'DSA  [YES=1, NO=2]_1.0', 'DSA  [YES=1, NO=2]_2.0'
# Note 2: In categorical variables, we have dropped the first dummy variable to reduce the risk of multicollinearity in models

# Plot the feature importance histogram based on AUC-ROC scores with error bars

plt.figure(figsize=(12, 9))

#define color for the bars
color = "#89CFF0"
#plot the bars
ax = sns.barplot(y='AUC-ROC', x='Feature', data=auc_importance_df, color=color, errorbar=None, linewidth=2)

# Add error bars manually
for i, (importance, error) in enumerate(zip(auc_importance_df['AUC-ROC'], auc_importance_df['Error'])):
    plt.errorbar(i, importance, yerr=error, fmt='none', ecolor='black', capsize=5, linewidth=2)

# Allow user to pick x-ticks manually
selected_features_auc = auc_importance_df['Feature'].tolist()  # Use mapped feature names

##Set limit to y-axis
# plt.ylim(0.01, None)  # Set lower limit to 0.4 and upper limit to automatic

plt.xticks(ticks=range(len(selected_features_auc)), labels=selected_features_auc, rotation=90, fontsize=18, fontname='Arial', fontweight='bold')
plt.ylabel('Importance (AUC-ROC)', fontsize=32, fontname='Arial', fontweight='bold')
plt.xlabel('Feature', fontsize=32, fontname='Arial', fontweight='bold')
plt.yticks(fontsize=26, fontname='Arial', fontweight='bold')
plt.tight_layout()

# # Add horizontal line at y = 0.5
# plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2)

# Increase linewidth of the outer plot box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)  # Adjust the linewidth as needed

# Save and display the plot
output_filename = desktop_path + 'FEATURE_IMPORTANCE_HISTOGRAM.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

# Create a DataFrame to store feature names, mean AUC scores, and errors
export_df = pd.DataFrame({
    'Feature': auc_importance_df['Feature'],
    'Mean AUC-ROC': auc_importance_df['AUC-ROC'],
    'Error': auc_importance_df['Error']
}).set_index('Feature').transpose()

# Define the output filename for Excel
excel_output_filename = desktop_path + 'FEATURE_IMPORTANCE_DATA.xlsx'

# Save the DataFrame to Excel
export_df.to_excel(excel_output_filename, index=True)

################# Produce ROC curves and the equivalent AUC-ROC for different models with different features based on their importance

# Define features for model with EB with the top 5 features
X_EB_withtop5 = data_encoded[['EB: CXCL10', 'EB: CXCL9', 'RECIPIENT AGE (YEARS)', 'DSA  [YES=1, NO=2]_2.0','Donor Age (Years)']]

# Define features for model with EB without the bottom 6 features/ with only the top 15 features
X_EB_withtop15 = data_encoded.drop(columns=['ELISA: CXCL10', 'ELISA: CXCL9', 'HLA-A [mm]', 'HLA-B [mm]',
                                           'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_3.0',
                                           'Serum Cr at time of biopsy [μmol/L]',
                                           'Donor Type [1=LIVE, 2=DBD, 3=DCD]_3.0',
                                           'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_4.0'])

# Define features for model with EB without the bottom 11 features/with the top 10 features
X_EB_withtop10 = data_encoded.drop(columns=['ELISA: CXCL10', 'ELISA: CXCL9', 'HLA-A [mm]', 'HLA-B [mm]',
                                           'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_3.0',
                                           'Serum Cr at time of biopsy [μmol/L]',
                                           'Donor Type [1=LIVE, 2=DBD, 3=DCD]_3.0',
                                           'Total Number of HLA mismatches',
                                           'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)_4.0',
                                           'Urinary PCR at time of biopsy [g/g]',
                                           'Urinary Creatinine at time of biopsy [mmol/L]',
                                           # 'Bacteruiria (>10,000cfu/ml) near time of biospy [YES=1, NO=2]_2.0',
                                           ])

#1. Plot the ROC curve and Youden index point: EB without the bottom 5 features
MODEL_EB_withtop15 = calculate_youden_and_plot(
    X_EB_withtop15, data[target_column], desktop_path + 'ROC_Curve_X_EB_withtop15.png',
    title="EB - With best 15 feat."
)

#2. Plot the ROC curve and Youden index point: EB without the bottom 10 features
MODEL_EB_withtop10 = calculate_youden_and_plot(
    X_EB_withtop10, data[target_column], desktop_path + 'ROC_Curve_X_EB_withtop10.png',
    title="EB - With best 10 feat."
)

#3. Plot the ROC curve and Youden index point: EB with the top 5 features
MODEL_EB_withtop5 = calculate_youden_and_plot(
    X_EB_withtop5, data[target_column], desktop_path + 'ROC_Curve_X_EB_withtop5.png',
    title="EB - With top 5 feat."
)


#### 4 Plot the Average ROC Curves and Youden Index Points for the Full Model using EB after Splitting into Training and Testing with Repeated Stratified K-Fold (n_splits=4, n_repeats=1000)

# Function to calculate average ROC curve using Repeated Stratified K-Fold
def calculate_avg_roc_with_repeated_kfold(X, y, n_splits, n_repeats, save_path, title):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # Initialize variables for storing metrics
    tprs_train, tprs_test = [], []
    aucs_train, aucs_test = [], []
    mean_fpr = np.linspace(0, 1, 100)  # Define a common FPR range for interpolation

    iteration_count = 0
    test_auc_values_per_iteration = []  # Store test AUC for each iteration
    train_auc_values_per_iteration = []  # Store train AUC for each iteration

    # Iterate through folds
    for train_idx, test_idx in cv.split(X, y):

        # Increment the iteration count here
        iteration_count += 1

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the logistic regression model
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(X_train, y_train)

        # Compute ROC curve and AUC for training set
        y_probs_train = model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_probs_train)
        tprs_train.append(interp(mean_fpr, fpr_train, tpr_train))
        aucs_train.append(roc_auc_score(y_train, y_probs_train))
        train_auc = roc_auc_score(y_train, y_probs_train)

        # Compute ROC curve and AUC for testing set
        y_probs_test = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_probs_test)
        tprs_test.append(interp(mean_fpr, fpr_test, tpr_test))
        aucs_test.append(roc_auc_score(y_test, y_probs_test))
        test_auc = roc_auc_score(y_test, y_probs_test)

        # Store test AUC for this iteration
        test_auc_values_per_iteration.append((iteration_count, test_auc))
        train_auc_values_per_iteration.append((iteration_count, train_auc))

    # Compute mean and std for training and testing ROC curves
    mean_tpr_train = np.mean(tprs_train, axis=0)
    std_tpr_train = np.std(tprs_train, axis=0)
    mean_tpr_test = np.mean(tprs_test, axis=0)
    std_tpr_test = np.std(tprs_test, axis=0)

    # Calculate mean and std AUC
    mean_auc_train = np.mean(aucs_train)
    std_auc_train = np.std(aucs_train)
    mean_auc_test = np.mean(aucs_test)
    std_auc_test = np.std(aucs_test)

    # Plot average ROC curves
    plt.figure(figsize=(12, 9))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3)  # Random model line

    # Training ROC curve
    plt.plot(mean_fpr, mean_tpr_train, label=f'Training (AUC = {mean_auc_train:.3f} ± {std_auc_train:.3f})', color='blue', linewidth=4)
    plt.fill_between(mean_fpr, mean_tpr_train - std_tpr_train, mean_tpr_train + std_tpr_train, color='blue', alpha=0.2)

    # Testing ROC curve
    plt.plot(mean_fpr, mean_tpr_test, label=f'Testing (AUC = {mean_auc_test:.3f} ± {std_auc_test:.3f})', color='green', linewidth=4)
    plt.fill_between(mean_fpr, mean_tpr_test - std_tpr_test, mean_tpr_test + std_tpr_test, color='green', alpha=0.2)

    # Enhance the plot
    plt.xlabel('False Positive Rate', fontsize=32, fontweight='bold', labelpad=15)
    plt.ylabel('True Positive Rate', fontsize=32, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=26, fontweight='bold')
    plt.yticks(fontsize=26, fontweight='bold')
    legend = plt.legend(loc='lower right', fontsize=25, frameon=False)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Save the plot
    plt.savefig(save_path)
    plt.close()

    # Save the average AUC scores and standard deviation
    metrics = {
        "Set": ["Training", "Testing"],
        "Mean AUC": [mean_auc_train, mean_auc_test],
        "Std AUC": [std_auc_train, std_auc_test],
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel(save_path.replace(".png", ".xlsx"), index=False)

    return mean_auc_test, std_auc_test, test_auc_values_per_iteration, mean_auc_train, std_auc_train, train_auc_values_per_iteration


# Apply the function for EB with all the features
avg_auc_test, std_auc_test, test_auc_values, avg_auc_train, std_auc_train, train_auc_values = calculate_avg_roc_with_repeated_kfold(
    X_full_EB,
    data[target_column],
    n_splits=4,
    n_repeats=100,
    save_path=desktop_path + 'ROC_Curve_EB_FULL_MODEL_avg_repeated.png',
    title="EB - Full Model (Avg ROC with Repeated K-Fold)",
)

#you can print the following: "Avg AUC (Testing): 0.802, Std AUC (Testing): 0.127"
# print(f"Avg AUC (Testing): {avg_auc_test:.3f}, Std AUC (Testing): {std_auc_test:.3f}")

# Plot the AUC-ROC for each iteration for the test set
iterations = [val[0] for val in test_auc_values]
test_aucs = [val[1] for val in test_auc_values]
plt.figure(figsize=(12, 8))
plt.plot(iterations, test_aucs, marker='o', linewidth=2, color='purple')
# Set the new y-limits from 0 to 1.02
plt.ylim(0, 1.02)
plt.xlabel('Iteration', fontsize=32, fontweight='bold', labelpad=15)
plt.ylabel('Test AUC-ROC', fontsize=32, fontweight='bold', labelpad=15)
plt.xticks(fontsize=26, fontweight='bold')
plt.yticks(fontsize=26, fontweight='bold')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
# Horizontal line for mean AUC
mean_auc_line = np.mean(test_aucs)
plt.axhline(y=mean_auc_line, color='red', linestyle='--', linewidth=3,
            label=f'Mean AUC: {mean_auc_line:.3f}')
plt.legend(fontsize=50, frameon=False)
for text in plt.legend().get_texts():
    text.set_fontweight('bold')
plt.savefig(desktop_path + 'Test_AUC_iterations_line_plot.png', dpi=300, bbox_inches='tight')
plt.close()



# Plot the AUC-ROC for each iteration for training set
iterations_train = [val[0] for val in train_auc_values]
train_aucs = [val[1] for val in train_auc_values]
plt.figure(figsize=(12, 8))
plt.plot(iterations_train, train_aucs, marker='o', linewidth=2, color='purple')
# Set the new y-limits from 0 to 1.05
plt.ylim(0, 1.05)
plt.xlabel('Iteration', fontsize=32, fontweight='bold', labelpad=15)
plt.ylabel('Train AUC-ROC', fontsize=32, fontweight='bold', labelpad=15)
plt.xticks(fontsize=26, fontweight='bold')
plt.yticks(fontsize=26, fontweight='bold')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
# Horizontal line showing the mean AUC for training
mean_auc_line_train = np.mean(train_aucs)
plt.axhline(y=mean_auc_line_train, color='red', linestyle='--', linewidth=3, label=f'Mean AUC: {mean_auc_line_train:.3f}')
plt.legend(fontsize=30, frameon=False)
for text in plt.legend().get_texts():
    text.set_fontweight('bold')
plt.savefig(desktop_path + 'Train_AUC_iterations_line_plot.png', dpi=300, bbox_inches='tight')
plt.close()

############################################################################################



########################  PART 3: Calculate the Pearson Correlation Coefficient Matrix and plot it as a heatmap, and save the data ########################

# Encode categorical features that are binary by removing the excess dummy variables
binary_categorical_features = [
    'RECIPIENT SEX [1=M, 2=F]',
    'DSA  [YES=1, NO=2]',
    # 'Bacteruiria (>10,000cfu/ml) near time of biopsy [YES=1, NO=2]',  # Uncomment if needed
]

categorical_features_encoded_PCC_binary = pd.get_dummies(
    data[binary_categorical_features],
    drop_first=True
)

# Encode categorical features that are non-binary by not removing the excess dummy variables
non_binary_categorical_features = [
    'Recipient Ethnicity (1=White, 2=Asian, 3=Afro-carribean, 4=Other)',
    'Donor Type [1=LIVE, 2=DBD, 3=DCD]'
]

categorical_features_encoded_PCC_non_binary = pd.get_dummies(
    data[non_binary_categorical_features],
    drop_first=False
)

# Combine encoded categorical features
categorical_features_encoded_PCC = pd.concat(
    [categorical_features_encoded_PCC_non_binary, categorical_features_encoded_PCC_binary],
    axis=1
)

# Combine numerical features and encoded categorical features with the target column
data_encoded_PCC = pd.concat(
    [data[numerical_features], categorical_features_encoded_PCC],
    axis=1
)
data_encoded_PCC = pd.concat(
    [data[target_column], data_encoded_PCC],
    axis=1
)

# Rename the target column to 'Acute Rejection'
data_encoded_PCC.rename(columns={target_column: 'Acute Rejection'}, inplace=True)

# Rename all columns in data_encoded_PCC using feature_name_mapping
data_encoded_PCC.rename(columns=feature_name_mapping, inplace=True)

# Verify that the columns have been renamed correctly
print("Columns in data_encoded_PCC after renaming:")
print(data_encoded_PCC.columns.tolist())

# Compute the Pearson Correlation Coefficient Matrix (absolute values) for all features
PCC_all = data_encoded_PCC.corr().abs()

# No need to remap since columns are already renamed
mapped_feature_names_all = [feature_name_mapping(feature) for feature in PCC_all.columns]

# Update the column and index names of PCC matrix to mapped feature names
# This step is redundant but kept for consistency
PCC_all.columns = mapped_feature_names_all
PCC_all.index = mapped_feature_names_all

# Plot the PCC matrix for all features
plt.figure(figsize=(20, 18))  # Adjusted figure size for better readability

# Generate a custom colormap
cmap = sns.color_palette("YlGnBu", as_cmap=True)

# Generate a mask for the upper triangle
mask_all = np.triu(np.ones_like(PCC_all, dtype=bool), k=1)

# Draw the heatmap with the mask and custom labels
heatmap_all = sns.heatmap(
    PCC_all,
    mask=mask_all,
    cmap=cmap,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 12, "weight": "bold"},  # Adjusted annotation font size for readability
    square=True,
    cbar_kws={"shrink": .5, "label": "Correlation"},
    xticklabels=mapped_feature_names_all,
    yticklabels=mapped_feature_names_all,
    vmin=0,
    vmax=1,
)

# Set colorbar label font size, add padding, and make it bold
colorbar_all = heatmap_all.collections[0].colorbar
colorbar_all.ax.tick_params(labelsize=14, width=1)  # Adjust colorbar ticks
colorbar_all.set_label('Correlation', fontsize=14, weight='bold', labelpad=10)  # Increase padding with labelpad

# Manually set colorbar tick labels to bold
for label in colorbar_all.ax.get_yticklabels():
    label.set_fontweight('bold')

# Set tick labels font size and make them bold
plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')  # Adjusted rotation for better readability
plt.yticks(fontsize=16, fontweight='bold')  # Adjusted y-tick font size

# Set axis labels and make them bold
plt.xlabel('Features', fontsize=20, fontweight='bold')
plt.ylabel('Features', fontsize=20, fontweight='bold')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Save the full PCC heatmap to a TIFF file
output_filename_all = os.path.join(desktop_path, 'PCC_All_Features.tiff')
plt.savefig(output_filename_all, format='tiff', dpi=300, bbox_inches='tight')
plt.close()

# Optionally, save the full PCC matrix to an Excel file
excel_output_filename_all = os.path.join(desktop_path, 'PCC_All_Features_matrix.xlsx')
PCC_all.to_excel(excel_output_filename_all, index=True)

print(f"\nPCC heatmap for all features saved to {output_filename_all}")
print(f"PCC matrix for all features saved to {excel_output_filename_all}")

# ---------------------------------------------
# Select Top 10 Features Based on Correlation with 'Acute Rejection'
# ---------------------------------------------

# Calculate correlation with 'Acute Rejection' and select top 10 features
correlation_with_target = PCC_all['Acute Rejection'].drop('Acute Rejection')  # Exclude self-correlation
top10_features = correlation_with_target.sort_values(ascending=False).head(10).index.tolist()

print("\nTop 10 features based on correlation with Acute Rejection:")
for i, feature in enumerate(top10_features, start=1):
    print(f"{i}. {feature} (Correlation: {correlation_with_target[feature]:.2f})")

# Include 'Acute Rejection' to have 11 features in total
selected_features_for_pcc = ['Acute Rejection'] + top10_features

# Extract the data for these features
data_selected_pcc = data_encoded_PCC[selected_features_for_pcc]

# Compute the Pearson Correlation Coefficient Matrix (absolute values) for selected features
PCC_top10 = data_selected_pcc.corr().abs()

# Reorder the columns and rows to match the selected_features_for_pcc order
# This ensures that 'Acute Rejection' is first, followed by top features sorted by correlation
PCC_top10 = PCC_top10.loc[selected_features_for_pcc, selected_features_for_pcc]

# ---------------------------------------------
# Compute and Plot PCC for Top 10 Features
# ---------------------------------------------

# Create a list of mapped feature names in the desired order
mapped_feature_names_top10 = [feature_name_mapping(feature) for feature in selected_features_for_pcc]

# Update the column and index names of PCC matrix to mapped feature names
PCC_top10.columns = mapped_feature_names_top10
PCC_top10.index = mapped_feature_names_top10

# Plot the PCC matrix for top 10 features
plt.figure(figsize=(15, 13))  # Adjusted figure size for top 10 features

# Generate a custom colormap
cmap = sns.color_palette("YlGnBu", as_cmap=True)

# Generate a mask for the upper triangle
mask_top10 = np.triu(np.ones_like(PCC_top10, dtype=bool), k=1)

# Draw the heatmap with the mask and custom labels
heatmap_top10 = sns.heatmap(
    PCC_top10,
    mask=mask_top10,
    cmap=cmap,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 14.5, "weight": "bold"},  # Adjusted annotation font size for readability
    square=True,
    cbar_kws={"shrink": .8, "label": "Correlation"},
    xticklabels=mapped_feature_names_top10,
    yticklabels=mapped_feature_names_top10,
    vmin=0,
    vmax=1,
)

# Set colorbar label font size, add padding, and make it bold
colorbar_top10 = heatmap_top10.collections[0].colorbar
colorbar_top10.ax.tick_params(labelsize=16, width=1)  # Adjust colorbar ticks
colorbar_top10.set_label('Correlation', fontsize=16, weight='bold', labelpad=10)  # Increase padding with labelpad

# Manually set colorbar tick labels to bold
for label in colorbar_top10.ax.get_yticklabels():
    label.set_fontweight('bold')

# Set tick labels font size and make them bold
plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')  # Adjusted rotation for better readability
plt.yticks(fontsize=16, fontweight='bold')  # Adjusted y-tick font size

# Set axis labels and make them bold
plt.xlabel('Features', fontsize=23.5, fontweight='bold')
plt.ylabel('Features', fontsize=23.5, fontweight='bold')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Save the top 10 PCC heatmap to a TIFF file
output_filename_top10 = os.path.join(desktop_path, 'PCC_Top10_Features.tiff')
plt.savefig(output_filename_top10, format='tiff', dpi=300, bbox_inches='tight')
plt.close()

# Optionally, save the PCC matrix for top 10 features to an Excel file
excel_output_filename_top10 = os.path.join(desktop_path, 'PCC_Top10_Features_matrix.xlsx')
PCC_top10.to_excel(excel_output_filename_top10, index=True)

print(f"PCC heatmap for top 10 features based on correlation to Acute Rejection saved to {output_filename_top10}")
print(f"PCC matrix for top 10 features saved to {excel_output_filename_top10}")

#############################################################################
##########################################################################





################################## Part4: p-values and ORs for all the features #############################

# Helper function to extract p-values and calculate ORs for each feature (univariate)
def get_pvalues_or(X, y, model_name):
    """
    Returns a DataFrame with Feature, Coefficient, p-value, and Odds Ratio
    for univariate logistic regression on each feature in X.
    """
    results_list = []
    for feature in X.columns:
        # Prepare the data for univariate logistic regression
        X_feature = sm.add_constant(X[[feature]], has_constant='add')
        model = sm.Logit(y, X_feature)
        result = model.fit(disp=False)  # disp=False suppresses console output

        coef = result.params[feature]
        pval = result.pvalues[feature]
        or_val = np.exp(coef)  # odds ratio

        results_list.append({
            'Feature': feature,
            'Coefficient': coef,
            'p-value': pval,
            'Odds Ratio': or_val
        })

    results_df = pd.DataFrame(results_list).sort_values(by='p-value')
    print(f"\nUnivariate Logistic Regression Results for {model_name}:")
    print(results_df)
    return results_df

# Assuming 'y' is your target column from the DataFrame 'data'
y = data[target_column]

# Calculate p-values and ORs for:
df_pvals_full_ELISA = get_pvalues_or(X_full_ELISA, y, "Full Model ELISA")
df_pvals_full_EB = get_pvalues_or(X_full_EB, y, "Full Model EB")

# Define desktop path to save the data
desktop_path = "C:/Users/homeuser/Desktop/Paper_data/"

# Save the univariate results (p-values & ORs) to Excel
output_excel_path = desktop_path + "Univariate_Logistic_Results.xlsx"
with pd.ExcelWriter(output_excel_path) as writer:
    df_pvals_full_ELISA.to_excel(writer, sheet_name="Full Model ELISA", index=False)
    df_pvals_full_EB.to_excel(writer, sheet_name="Full Model EB", index=False)

print(f"\nAll univariate results (p-values and ORs) saved to {output_excel_path}")

###################### ###################

def get_pvalues_or_with_ci(X, y, model_name):
    """
    Returns a DataFrame with Feature, Coefficient, p-value, Odds Ratio, and 95% CI
    for univariate logistic regression on each feature in X.
    """
    results_list = []
    for feature in X.columns:
        # Prepare the data for univariate logistic regression
        X_feature = sm.add_constant(X[[feature]], has_constant='add')
        model = sm.Logit(y, X_feature)
        result = model.fit(disp=False)  # disp=False suppresses console output

        coef = result.params[feature]
        pval = result.pvalues[feature]
        or_val = np.exp(coef)  # odds ratio
        se = result.bse[feature]  # standard error of the coefficient

        # Calculate 95% confidence intervals
        lower_ci = coef - 1.96 * se
        upper_ci = coef + 1.96 * se
        or_lower_ci = np.exp(lower_ci)  # CI for the odds ratio
        or_upper_ci = np.exp(upper_ci)

        results_list.append({
            'Feature': feature,
            'Coefficient': coef,
            'p-value': pval,
            'Odds Ratio': or_val,
            'OR 95% CI (Lower)': or_lower_ci,
            'OR 95% CI (Upper)': or_upper_ci
        })

    results_df = pd.DataFrame(results_list).sort_values(by='p-value')
    print(f"\nUnivariate Logistic Regression Results with 95% CI for {model_name}:")
    print(results_df)
    return results_df

df_pvals_full_with_ci_ELISA = get_pvalues_or_with_ci(X_full_ELISA, y, "Full Model ELISA")
df_pvals_full_with_ci_EB = get_pvalues_or_with_ci(X_full_EB, y, "Full Model EB")

# Save the univariate results (p-values & ORs) to Excel
output_excel_path = desktop_path + "Univariate_Logistic_Results_with_CI.xlsx"
with pd.ExcelWriter(output_excel_path) as writer:
    df_pvals_full_with_ci_ELISA.to_excel(writer, sheet_name="Full Model ELISA", index=False)
    df_pvals_full_with_ci_EB.to_excel(writer, sheet_name="Full Model EB", index=False)

print(f"\nAll univariate results (p-values and ORs) saved to {output_excel_path}")

# Assuming `df_pvals_full_ELISA` is the DataFrame with the results from the univariate logistic regression
# and contains a column `Feature` for the feature names and a column `p-value` for their corresponding p-values.

# Sort features by p-value for better visualization and select the top 10 features with smallest p-values
df_pvals_top10 = df_pvals_full_ELISA.sort_values(by='p-value').head(10)

# Apply the feature_name_mapping to the top 10 features
df_pvals_top10['Feature'] = df_pvals_top10['Feature'].map(feature_name_mapping)

# Plot the p-values for the top 10 features with mapped names
plt.figure(figsize=(12, 8))

# Define color for the bars
color = "#89CFF0"

# Plot the bars
ax = sns.barplot(
    x='Feature',
    y='p-value',
    data=df_pvals_top10,
    color=color,
    linewidth=2
)

# Customize the plot aesthetics
plt.xticks(rotation=90, fontsize=16, fontname='Arial', fontweight='bold')
plt.ylabel('p-value', fontsize=26, fontname='Arial', fontweight='bold')
plt.xlabel('Feature', fontsize=28, fontname='Arial', fontweight='bold')
plt.yticks(fontsize=14, fontname='Arial', fontweight='bold')
plt.tight_layout()

# Add a horizontal line at the 0.05 threshold for statistical significance
plt.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (p=0.05)')

# Enhance legend appearance
plt.legend(fontsize=18, frameon=False)
for text in plt.legend().get_texts():
    text.set_fontweight('bold')

# Increase linewidth of the outer plot box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# Save and display the plot
output_filename = desktop_path + "Bar_Plot_P_Values_ELISA_Top10.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBar plot of top 10 p-values with mapped feature names saved to {output_filename}")

#############################################################################
################# Extra calculations: Lin's Concordance Correlation Coefficient (CCC) #########################################

##########################################################################


def lins_ccc(x, y):
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC) between two arrays x and y.
    Reference:
    Lin, L. I. (1989). A Concordance Correlation Coefficient to Evaluate Reproducibility.
    Biometrics, 45(1), 255-268.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Means
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Variances
    var_x = np.var(x, ddof=1)  # sample variance
    var_y = np.var(y, ddof=1)

    # Standard deviations
    sd_x = np.sqrt(var_x)
    sd_y = np.sqrt(var_y)

    # Pearson correlation coefficient
    r_xy = np.corrcoef(x, y)[0, 1]

    # Lin's CCC formula
    ccc = (2 * r_xy * sd_x * sd_y) / (var_x + var_y + (mean_x - mean_y) ** 2)
    return ccc


# Compute Lin's CCC for CXCL9
# Make sure to drop rows with NaN values in these two columns:
cxcl9_nonan = data[['ELISA: CXCL9', 'EB: CXCL9']].dropna()
ccc_cxcl9 = lins_ccc(cxcl9_nonan['ELISA: CXCL9'], cxcl9_nonan['EB: CXCL9'])

# Compute Lin's CCC for CXCL10
cxcl10_nonan = data[['ELISA: CXCL10', 'EB: CXCL10']].dropna()
ccc_cxcl10 = lins_ccc(cxcl10_nonan['ELISA: CXCL10'], cxcl10_nonan['EB: CXCL10'])

print(f"Lin's CCC for CXCL9 (ELISA vs. EB): {ccc_cxcl9:.3f}")
print(f"Lin's CCC for CXCL10 (ELISA vs. EB): {ccc_cxcl10:.3f}")
