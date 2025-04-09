# Clinically-Relevant-Detection-of-Kidney-Transplant-Acute-Rejection-publication-cxcl-analysis
A rapid, label‐free electrochemical biosensor detects urinary CXCL9/10 at pg/mL levels in under 15 minutes. Using a MXene/BSA hydrogel–modified screen‐printed electrode, it accurately identifies kidney transplant acute rejection, offering a fast, non‐invasive alternative to biopsies.

# README for Clinical Data Analysis and ROC-AUC Workflow

## Overview
This repository provides a comprehensive workflow for analyzing clinical data to evaluate model performance, feature importance, and relationships between features. The workflow includes calculations of ROC curves, Youden indices, Pearson correlation matrices, feature importance, and univariate logistic regression analysis. Visualizations and outputs are saved in structured formats for further use.

## Repository Structure
- `clinical_data.xlsx`: The clinical dataset containing patient and biopsy-related data.
- `main.py`: The Python script that performs the analyses described below.
- `Paper_data/`: Directory where all output files (plots, Excel files, etc.) are saved.

## Workflow Details

### 1. Calculate ROC Curves and Youden Indices
#### Purpose:
- Evaluate the performance of logistic regression models by plotting ROC curves.
- Identify the optimal threshold using Youden's index.

#### Steps:
1. Models were trained using logistic regression on various feature combinations:
   - **Full Model (ELISA)**: Includes all features except EB measurements.
   - **Full Model (EB)**: Includes all features except ELISA measurements.
   - **ELISA Only**: Uses only ELISA CXCL10 and CXCL9 features.
   - **EB Only**: Uses only EB CXCL10 and CXCL9 features.
2. Outputs:
   - ROC curves saved as PNG files.
   - Youden indices, optimal thresholds, TPR, and FPR saved in `Youden_Indices.xlsx`.
   - Confusion matrices saved in `Confusion_Matrices.xlsx`.

#### Key Functions:
- `calculate_youden_and_plot`: Computes ROC curves, Youden indices, and performance metrics.

#### Output Files:
- `ROC_Curve_FULL_MODEL_ELISA.png`
- `ROC_Curve_FULL_MODEL_EB.png`
- `ROC_Curve_ELISA_ONLY.png`
- `ROC_Curve_EB_ONLY.png`
- `Youden_Indices.xlsx`
- `Confusion_Matrices.xlsx`

### 2. Feature Importance Based on AUC-ROC Scores
#### Purpose:
- Determine the importance of individual features by calculating AUC-ROC scores using logistic regression.
- Perform cross-validation to ensure robust evaluation.

#### Steps:
1. Each feature was evaluated individually using repeated stratified k-fold cross-validation.
2. AUC scores and errors were plotted in a bar chart.
3. Results were saved in `FEATURE_IMPORTANCE_DATA.xlsx`.

#### Key Functions:
- `RepeatedStratifiedKFold`: Used for cross-validation.
- `LogisticRegression`: Used for classification.

#### Output Files:
- `FEATURE_IMPORTANCE_HISTOGRAM.png`
- `FEATURE_IMPORTANCE_DATA.xlsx`

### 3. Pearson Correlation Coefficient Matrix
#### Purpose:
- Analyze relationships between features using Pearson correlation coefficients.
- Identify the top correlated features with the target variable (Acute Rejection).

#### Steps:
1. The full correlation matrix was computed and visualized as a heatmap.
2. The top 10 features most correlated with the target variable were selected for further analysis.
3. Results were saved in Excel files.

#### Output Files:
- `PCC_All_Features.tiff`: Heatmap for all features.
- `PCC_All_Features_matrix.xlsx`: Full correlation matrix.
- `PCC_Top10_Features.tiff`: Heatmap for the top 10 features.
- `PCC_Top10_Features_matrix.xlsx`: Correlation matrix for the top 10 features.

### 4. Univariate Logistic Regression (P-values and Odds Ratios)
#### Purpose:
- Evaluate the significance of individual features using p-values.
- Calculate Odds Ratios (ORs) and confidence intervals for each feature.

#### Steps:
1. Univariate logistic regression was performed for each feature.
2. Results include coefficients, p-values, ORs, and 95% confidence intervals.
3. Results were plotted and saved in Excel files.

#### Key Functions:
- `get_pvalues_or`: Calculates p-values and ORs.
- `get_pvalues_or_with_ci`: Extends `get_pvalues_or` to include confidence intervals.

#### Output Files:
- `Univariate_Logistic_Results.xlsx`: P-values and ORs for all features.
- `Univariate_Logistic_Results_with_CI.xlsx`: Includes confidence intervals.
- `Bar_Plot_P_Values_ELISA_Top10.png`: Visualization of top 10 features based on p-values.

### 5. Additional Visualizations
#### Purpose:
- Compare ROC curves for different models.
- Plot AUC-ROC scores for iterations during cross-validation.

#### Output Files:
- `ROC_Curve_FULL_MODEL_ELISA_vs_EB.png`
- `Test_AUC_iterations_line_plot.png`
- `Train_AUC_iterations_line_plot.png`

## Installation and Usage
### Prerequisites
- Python 3.8 or later
- Required Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `statsmodels`

### Setup
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `clinical_data.xlsx` is in the repository.

### Running the Script
1. Execute the script using:
   ```bash
   python main.py
   ```
2. Follow the prompts to select `clinical_data.xlsx` when prompted.

### Outputs
All results are saved in the `Paper_data/` directory. Ensure the directory exists or is created automatically by the script.

## Interpretation of Results
- **ROC Curves**: Provide insights into model discrimination performance.
- **Youden Indices**: Help determine optimal thresholds for binary classification.
- **Feature Importance**: Highlights the most predictive features.
- **Correlation Matrices**: Show relationships between features and the target variable.
- **Univariate Logistic Regression**: Identifies statistically significant features and their impact (ORs).

## Customization
You can modify the following sections:
- **Feature selection**: Adjust the `features` list in the script.
- **Models**: Use different classifiers by replacing `LogisticRegression`.
- **Visualization**: Customize plots by changing styles or annotations.

