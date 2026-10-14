# AI HR Workforce Optimization System

An AI-driven Human Resources analysis and workforce optimization tool built in Python. This application analyzes employee attrition and behavior patterns to predict layoff risks, department-level hiring needs, and automatically generates an executive PDF report with visual charts.

---

## 🚀 Features

- **Tkinter Desktop GUI**: Simple interface for uploading HR datasets and exporting reports.
- **Machine Learning Predictive Model**:
  - Uses `scikit-learn` Logistic Regression to predict layoff risk per employee.
  - Computes department-level risk scores to categorize areas requiring hiring or potential downsizing.
- **Data Visualization**: Generates a distribution pie chart of departments needing hiring.
- **Automated PDF Report**: Uses `reportlab` to compile a styled PDF report containing:
  - Formatted tables with color-coded risk levels.
  - Embedded chart visuals.
  - Executive summary and timestamp.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaikharisha481-coder/HR-ANALYSIS.git
   cd HR-ANALYSIS
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

---

## 📊 Dataset Requirements

The input dataset should be a `.csv` file containing:
- `department`: Categorical department names
- `salary`: Categorical salary bands (e.g., low, medium, high)
- `left`: Target column (1 for attrition/left, 0 for retained)
- Employee numerical feature columns used for attrition modeling
