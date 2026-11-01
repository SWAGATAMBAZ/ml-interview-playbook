# ML Interview Playbook

An interactive, single-stop site for machine-learning, data-science and MLOps interview preparation.

**Live site:** https://swagatambaz.github.io/ml-interview-playbook/

## What's inside

- **Foundations:** ML fundamentals, data prep and feature engineering, evaluation metrics, cross-validation and tuning, improving models and explainability, statistics and A/B testing.
- **13 model deep-dives:** linear regression, Ridge/Lasso/Elastic Net, logistic regression, kNN, Naive Bayes, SVM, decision trees, bagging and Random Forest, AdaBoost and gradient boosting, XGBoost/LightGBM/CatBoost, K-means, hierarchical/DBSCAN/GMM and anomaly detection, PCA and dimensionality reduction, and time-series forecasting.
  Each page covers a 30-second interview answer, intuition, the maths, assumptions, an interactive demo, hyperparameters, how to improve the model, pros and cons, code, auto-graded labs and interview Q&A.
- **Production:** MLOps (serving, deployment strategies, CI/CD/CT, monitoring, drift with a PSI demo, retraining, incident playbook) and ML system design with four worked case studies.
- **Practice:**
  - ~300 interview questions with model answers. You can browse them, study them as flashcards, or take a timed mock interview.
  - 76 runnable Python labs and coding drills, including a capstone.
  - A shared-state Python playground.
  - A printable cheat sheet with an interactive model chooser.

## How the labs work

Python runs in the browser through [Pyodide](https://pyodide.org) (v0.29.5) inside a Web Worker, so nothing needs installing. NumPy, pandas, scikit-learn, SciPy, statsmodels, matplotlib, XGBoost and LightGBM are available. Each lab has starter code, hints, a solution and an automatic checker. Progress, code and “known” questions are saved in `localStorage`.

## Development

Everything is static HTML/CSS/JS, so no build step is needed to serve it. `build.py` regenerates the search index, the question bank and the labs index. It can also execute every lab against a local Python environment:

```bash
python3 build.py              # assign question IDs, write assets/qbank.js, search-index.js, labs-index.js
python3 build.py --test       # run every lab: solutions must pass their checks, starters must not
LAB_PY=/path/to/python python3 build.py --test drill-   # test a subset with a specific interpreter
```

For the checks to match the browser, the test environment should use the same versions as Pyodide 0.29.5: scikit-learn 1.7.0, pandas 2.3.3, NumPy 2.2.5, statsmodels 0.14.4, XGBoost 2.1.4 and LightGBM 4.6.0.

KaTeX and CodeMirror are vendored in `assets/vendor/`. Pyodide loads from the jsDelivr CDN.
