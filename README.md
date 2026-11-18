# ML Interview Playbook

An interactive, single-stop site for machine-learning, data-science and MLOps interview preparation.

**Live site:** https://swagatambaz.github.io/ml-interview-playbook/

## What's inside

- **Foundations:** ML fundamentals, data prep and feature engineering, evaluation metrics, cross-validation and tuning, improving models and explainability, statistics and A/B testing.
- **Model deep-dives:** linear regression, Ridge/Lasso/Elastic Net, logistic regression, kNN, Naive Bayes, SVM, K-means, hierarchical/DBSCAN/GMM and anomaly detection, PCA and dimensionality reduction.
- **Tree-based models (own section):** a family overview built for intuition (ten mental models, a five-model interactive comparison, extrapolation demo, comparison table), then decision trees, bagging, Random Forest & Extra Trees, AdaBoost (hand-worked example), gradient boosting, XGBoost, LightGBM and CatBoost.
- **Forecasting (own section, 7 pages):** fundamentals and backtesting; baselines, exponential smoothing, ETS, Theta, STL/MSTL/TBATS; ARIMA, SARIMA, SARIMAX, VAR, GARCH and Kalman filters; ML forecasting with lag features, global LightGBM, multi-step strategies, Fourier terms and Prophet; deep learning from RNN/LSTM/GRU to DeepAR, N-BEATS, TFT, PatchTST and foundation models; probabilistic, hierarchical and intermittent forecasting; and an improvement/bottleneck playbook with 30 situational questions.
  Each page goes from plain English to advanced intuition: a step-by-step explanation without maths, a 30-second interview answer, must-know points, advanced intuition, a “when it wins vs other models” section with a practical example, optional (collapsed) maths, hyperparameters, code, auto-graded labs and interview Q&A.
- **Production:** MLOps (serving, deployment strategies, CI/CD/CT, monitoring, drift with a PSI demo, retraining, incident playbook) and ML system design with four worked case studies.
- **Practice:**
  - ~300 interview questions with model answers. You can browse them, study them as flashcards, or take a timed mock interview.
  - 76 runnable Python labs and coding drills, including a capstone.
  - A shared-state Python playground.
  - A printable cheat sheet with an interactive model chooser.

## Added in the trees & forecasting update

- New pages: `trees.html`, `bagging.html`, `adaboost.html`, `forecast-classical.html`, `arima.html`, `forecast-ml.html`, `forecast-dl.html`, `forecast-advanced.html`, `forecast-playbook.html`.
- 10 new interactive demos (`assets/demos4.js`): five tree ensembles on one dataset, tree extrapolation, bagging variance, exponential smoothing tuner, ARIMA simulator with ACF/PACF, fan charts, backtest designer, windowing, LSTM memory vs RNN, Croston vs SES vs TSB.
- 35 new auto-graded labs (AdaBoost and bagging from scratch, PACF order identification, Ljung–Box, SARIMAX with promotions, leak-free lag features, recursive vs direct, global vs local, an LSTM cell in NumPy, conformal intervals, Croston, hierarchical reconciliation and more).

## Added in the depth update

- Every model page has a **Layman → advanced** section: plain-English explanation, the algorithm step by step, a worked example by hand, what advanced interviewers probe, how the questioning escalates, and common mistakes.
- 15 more interactive demos (least-squares squares, loss functions, optimiser paths, L1 vs L2 geometry, logistic training, tree and forest boundaries, AdaBoost rounds, Naive Bayes calculator, kernel trick, DBSCAN, dendrogram, ACF, XGBoost gain calculator, loss-curve doctor).
- `situational.html`: 100 scenario questions with detailed answers.

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
