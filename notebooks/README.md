# Notebooks

This folder contains the notebooks to process the data and run the housing price models. The `html` folder contains HTML versions of the notebooks if you want to read through them.

What's inside?

```
> tree .
.
├── DataWrangling.ipynb
├── GWR.ipynb
├── htmls
│   ├── DataWrangling.html
│   ├── GWR.html
│   ├── OLS.html
│   ├── SpatialErrorModel.html
│   ├── SpatialErrorModelRegimes.html
│   ├── SpatialLagModel.html
│   ├── SpatialLagModelRegimes.html
│   └── StepwiseOLS.html
├── OLS.ipynb
├── RandomForestRegression.ipynb
├── SpatialErrorModel.ipynb
├── SpatialErrorModelRegimes.ipynb
├── SpatialLagModel.ipynb
├── SpatialLagModelRegimes.ipynb
└── StepwiseOLS.ipynb

0 directories, 9 files
```

I recommend following this order:

1. Data Wrangling - you have to process the data first!

2. Non-Spatial models.

*  OLS - this method uses a simple ordinary least squared (OLS) regression to predict housing prices.
* Stepwise OLS - this method using AIC to determine which variables to include in the regression and then performs OLS regression.
* Random Forest Regression - this is a common machine learning algorithm that uses regression. [To learn more see the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

3. Spatial models

* Spatial Error Model - [docs](https://spreg.readthedocs.io/en/latest/generated/spreg.ML_Error.html)
* Spatial Lag Model - [docs](https://spreg.readthedocs.io/en/latest/generated/spreg.ML_Lag.html)
* Spatial Error Model with Regimes - [docs](https://spreg.readthedocs.io/en/latest/generated/spreg.ML_Error_Regimes.html)
* Spatial Lag Model with Regimes - [docs](https://spreg.readthedocs.io/en/latest/generated/spreg.ML_Lag_Regimes.html)
* Geographically Weighted Regression (GWR) - [docs](https://pysal.org/notebooks/model/mgwr/intro.html)
