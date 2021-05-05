## Chicago Housing Models

This is a repository that focuses on applying multiple spatial and non-spatial models to Cook County residential sales. The `notebooks/` folder contains notebooks that walk through data wrangling and all of the models.


```
> tree .
.
├── data
│   ├── census
│   │   ├── ChicagoCensusTract.csv
│   │   └── DP03IL-CensusTract-Cleaned.csv
│   ├── Cook_County_Assessors_Residential_Sales_Data_2020.csv
│   ├── geodata
│   │   ├── Chicago_CensusTract.geojson
│   │   ├── Elementary_School_Tax_Districts-shp.zip
│   │   ├── High_School_Tax_District-shp.zip
│   │   ├── Illinois_CensusTract.geojson
│   │   ├── Park_Locations-shp
│   │   │   ├── Park_Locations.cpg
│   │   │   ├── Park_Locations.dbf
│   │   │   ├── Park_Locations.prj
│   │   │   ├── Park_Locations.shp
│   │   │   └── Park_Locations.shx
│   │   ├── Park_Locations-shp.zip
│   │   ├── Parks.geojson
│   │   ├── School_Locations-shp
│   │   │   ├── School_Locations.cpg
│   │   │   ├── School_Locations.dbf
│   │   │   ├── School_Locations.prj
│   │   │   ├── School_Locations.shp
│   │   │   └── School_Locations.shx
│   │   ├── School_Locations-shp.zip
│   │   └── Schools.geojson
│   └── Sales.tar
├── maps
│   ├── ChicagoSalePrice.html
│   └── ChicagoSalesCount.html
├── notebooks
│   ├── DataWrangling.ipynb
│   ├── GWR.ipynb
│   ├── OLS.ipynb
│   ├── RandomForestRegression.ipynb
│   ├── SpatialErrorModel.ipynb
│   ├── SpatialErrorModelRegimes.ipynb
│   ├── SpatialLagModel.ipynb
│   ├── SpatialLagModelRegimes.ipynb
│   └── StepwiseOLS.ipynb
├── README.md
├── ref
│   ├── A HOUSE PRICE VALUATION BASED ON THE RANDOM FOREST.pdf
│   ├── A mass appraisal assessment study using machine learning based on multiple.pdf
│   ├── Mass Appraisal Models of Real Estate in the 21st Century: A Systematic Literature Review.pdf
│   ├── Mass appraisal of residential apartments: An application of Random forest.pdf
│   └── MullainathanSpiess_2017_Machine.Learning:An.Applied.Econometric.Approach.pdf
├── requirements.txt
├── scripts
│   ├── height_tree_rfr.py
│   └── many_trial_random_forest.py
└── statshelper.py
```