# Data Analytics Portfolio
### Sam | NYC-Based Data Analyst

Welcome to my data analytics portfolio. This repository contains projects 
and exercises completed as part of a structured data analytics curriculum 
covering SQL, Python, and machine learning.

---

## Featured Project

### NYC Construction Cost Estimation Model
**[View Project](projects/nyc_construction_cost_estimation.ipynb)**

A machine learning model that predicts unit construction costs per gross 
square foot (GSF) for the 10 major trades on NYC construction projects, 
enabling data-driven bid preparation for general contractors.

**Business Problem**
General contractors in NYC rely heavily on individual estimator experience 
for bid preparation — a process prone to systematic errors including labor 
type misclassification and trade-specific underestimates. This model 
provides an objective, data-driven baseline for cost estimation.

**Technical Approach**
- Built and calibrated a synthetic dataset of 300 historical bids across 
  10 major trades, mirroring real NYC construction market rates
- Engineered features including trade type, labor type, project type, 
  GSF, and bid stage
- Trained and compared Linear Regression and Random Forest models
- Selected Linear Regression (R² = 0.9178, RMSE = 5.28/GSF) based on 
  superior performance at NYC construction project scale
- Built a configurable estimate_project() function that returns a full 
  cost breakdown by trade plus total project cost

**Key Finding**
Labor type is a significant cost driver in NYC construction — the model 
predicts a 13-15% cost premium for Union vs Open Shop labor across all 
project types, consistent with NYC prevailing wage requirements. When 
validated against an anonymized real NYC renovation project, labor type 
misclassification was identified as a primary source of budget variance.

**Tools Used**
Python, Pandas, scikit-learn, Matplotlib, Seaborn, Google Colab, GitHub

---

## Bootcamp Exercises
**[View Folder](bootcamp/)**

Structured weekly exercises covering the full data analytics stack:

| Week | Topic |
|---|---|
| Week 1 | Python Fundamentals |
| Week 2 | Introduction to Pandas |
| Week 3 | Pandas Intermediate |
| Week 4 | Pandas Advanced |
| Week 5 | SQL + Python Integration |
| Week 5b | Data Visualization |
| Week 5c | Real Data Practice (Titanic Dataset) |
| Week 6 | Statistical Foundations |
| Week 7 | Introduction to Machine Learning |
| Week 8 | scikit-learn Fundamentals |

---

## Skills
- **Languages:** Python, SQL
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Tools:** VS Code, Google Colab, GitHub, DB Browser for SQLite
- **Techniques:** Data cleaning, exploratory analysis, feature engineering, 
  regression modeling, classification modeling, data visualization

---

## In Progress
- Additional portfolio projects in construction and real estate analytics
- Machine learning fundamentals
- Advanced feature engineering

---
*This portfolio is actively maintained and updated as new projects are completed.*
