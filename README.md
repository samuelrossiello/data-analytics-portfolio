# Data Analytics Portfolio
### Sam | NYC-Based Data Analyst

Welcome to my data analytics portfolio. This repository contains machine 
learning, data analytics, and optimization projects built on real world 
domain expertise in NYC construction and management consulting, plus 
structured exercises from a Python and machine learning curriculum.

---

## Interactive Apps

### Construction Bid Optimizer — Interactive Tool
**[🚀 Launch Live App](https://construction-bid-optimizer.streamlit.app/)** | **[View Code](apps/construction_bid_optimizer_streamlit.py)**

Interactive web application for construction subcontractor buyout optimization — 
allows estimators to adjust project parameters, lock in preferred subcontractors 
based on relationship history, and instantly see optimized award recommendations 
and the cost impact of each preference decision.

**Live app works on any device — no installation required.**

**Features:**
- Adjustable GSF, MBE% and WBE% parameters
- Preferred subcontractor selection per trade
- Pure optimal vs greedy vs your selection comparison
- Three tier preference warning system (Good/Caution/Warning)
- Dynamic visualization charts
- Value of optimization metrics

**Tools:** Python, Pandas, PuLP, Matplotlib, Streamlit

---

---

## Projects

### Construction Bid Optimizer
**[View Project](projects/construction_bid_optimizer.ipynb)**

Linear programming model that identifies the optimal combination of 
subcontractor awards across 10 major trades — minimizing total buyout 
cost while meeting fixed MBE and WBE compliance requirements. Compared 
three strategies: greedy lowest bid (non-compliant), greedy lowest MWBE 
(compliant but inefficient), and optimized PuLP solution (compliant and 
optimal). The optimizer saved 2,358,596 vs the greedy MWBE approach on 
a single 500,000 GSF project by concentrating MWBE spend strategically 
rather than sequentially.

**Tools:** Python, Pandas, PuLP, Matplotlib, Google Colab, GitHub

---

### NYC Construction Cost Estimation Model
**[View Project](projects/nyc_construction_cost_estimation.ipynb)**

Predicts unit construction costs per GSF for the 10 major trades on NYC 
construction projects using Linear Regression (R² = 0.9178). Built a 
configurable estimation function returning full cost breakdowns by trade. 
Validated against an anonymized real NYC renovation project — labor type 
misclassification was identified as a primary source of budget variance.

**Tools:** Python, Pandas, scikit-learn, Matplotlib, Seaborn

---

### Consulting Project Risk Model
**[View Project](projects/consulting_project_risk_model.ipynb)**

Classifies consulting projects as at-risk before work begins using 
structured project characteristics and NLP analysis of contract text. 
Random Forest achieved 77% accuracy on structured features. Partner 
tenure was identified as the strongest predictor of project risk — 
nearly 3x more important than contract value — validating a U-shaped 
risk curve where junior and veteran partners carry the highest risk.

**Tools:** Python, Pandas, scikit-learn, TF-IDF, Matplotlib, Seaborn

---

## Skills
- **Languages:** Python, SQL
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, 
  TF-IDF Vectorizer, PuLP
- **Tools:** VS Code, Google Colab, GitHub, SQLite
- **Techniques:** Regression modeling, classification modeling, 
  NLP text analysis, linear programming optimization, feature 
  engineering, data visualization, exploratory analysis

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

## In Progress
- Machine learning fundamentals
- Advanced feature engineering
- Streamlit deployment of Construction Bid Optimizer

---
*This portfolio is actively maintained and updated as new projects are completed.*
