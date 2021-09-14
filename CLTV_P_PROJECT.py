##################################################################################################################
############################################# CLTV WITH SALES SAMPLE DATA ########################################
##################################################################################################################

##################################################################################################################
# Libraries
##################################################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################################################################################
# Reading Data
##################################################################################################################

data = pd.read_csv("Ders Notları/sales_data_sample.csv", encoding = "Latin-1")
df = data.copy()

##################################################################################################################
# Data Prepration
##################################################################################################################

df.shape
df.columns
df.info()
df.describe().T

# All the numeric variables are normally distributed

df.isnull().values.any()
df.isnull().sum()

df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])
df = df[~df["STATUS"].isin(["Cancelled", "Disputed"])]
df["TOTALSALES"] = df["SALES"] * df["QUANTITYORDERED"]

##################################################################################################################
# Identifying Varible Types
##################################################################################################################

## Categorical Variables

cat_cols = [i for i in df.columns if df[i].dtype == "O" and df[i].nunique() <= 20]

num_but_cat = [i for i in df.columns if df[i].dtype in [int, float] and df[i].nunique() <= 20]

cat_but_car = [i for i in df.columns if df[i].dtype == "O" and df[i].nunique() > 20]

## Numerical Variables

num_cols = [i for i in df.columns if df[i].dtype != "O" and i not in num_but_cat]

##################################################################################################################
# Analyzing and Data Visualization
##################################################################################################################

df.columns

df["DEALSIZE"].value_counts()

df["COUNTRY"].value_counts()

df["CITY"].value_counts()

df.groupby("YEAR_ID").agg({"SALES": ["mean", "sum"]})

year_month_sales = df.groupby(["YEAR_ID", "MONTH_ID"]).agg({"SALES": ["mean", "sum"]})
year_month_sales.to_excel("year_month_sales.xlsx")

plt.hist(df["SALES"])
plt.show()
plt.hist(df["QUANTITYORDERED"])
plt.show()

plt.boxplot(df["SALES"])
plt.show()
plt.boxplot(df["QUANTITYORDERED"])
plt.show()

df[["SALES", "QUANTITYORDERED"]].corr()

sns.lineplot(x = "YEAR_ID", y = "SALES", data = df)
plt.show()

df["YEAR-MONTH"] = df["YEAR_ID"].astype(str) + "-" + df["MONTH_ID"].astype(str)
line_df = df.groupby("YEAR-MONTH").agg({"SALES": "mean"})
line_df = line_df.reset_index()
line_df = line_df.sort_values(by = "YEAR-MONTH", ascending = True)
sns.lineplot(x = "YEAR-MONTH", y = "SALES", data = line_df, color = "green")
plt.xticks(fontsize = 14, rotation = 45)
plt.xlabel("Year and Month", fontsize = 14)
plt.ylabel("Sales Data", fontsize = 14)
plt.show()

##################################################################################################################
# Preprocessing
##################################################################################################################

def outlier_detector(df, var, low_quartile = 0.01, up_quartile = 0.99):
    quan1 = df[var].quantile(low_quartile)
    quan3 = df[var].quantile(up_quartile)

    interquartile_range = 1.5 * (quan3 - quan1)
    low_limit = interquartile_range - quan1
    up_limit = quan3 + interquartile_range

    return low_limit, up_limit

def outlier_destroyer(df, var):
    low_limit, up_limit = outlier_detector(df, var)

    df.loc[df[var] < low_limit, var] = low_limit
    df.loc[df[var] > up_limit, var] = up_limit

outlier_detector(df, "SALES")

outlier_destroyer(df, "SALES")

for i in [i for i in num_cols if i not in ["QTR_ID", "MONTH_ID", "YEAR_ID", "ORDERDATE"]]:
    outlier_destroyer(df, i)

##################################################################################################################
# Identfying CLTV_P Metrics
##################################################################################################################

cltv_df = df.groupby("CUSTOMERNAME").agg({"ORDERDATE": [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (df["ORDERDATE"].max() - x.min()).days],
                                          "ORDERNUMBER": lambda x: x.nunique(),
                                          "TOTALSALES": lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["Recency", "T", "Frequency", "Monetary"]

cltv_df.head()

cltv_df[cltv_df["Monetary"] > 1]
cltv_df[cltv_df["Frequency"] > 1]
cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]
cltv_df["T"] = cltv_df["T"] / 7
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df.head()

##################################################################################################################
# BG / NBD Model (Finding Expected Number of Transaction)
##################################################################################################################

bgf = BetaGeoFitter(penalizer_coef = 0.001)
bgf.fit(cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])

# Expected one week average sales order (first 10 customers)

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['Frequency'],
                                                        cltv_df['Recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# Adding a new variable that predicts average order sales by using BG/NBD model (expected one week sales)

cltv_df["expected_one_week_sales"] = bgf.predict(1,
                                                    cltv_df['Frequency'],
                                                    cltv_df['Recency'],
                                                    cltv_df['T'])

cltv_df.head()

# Expected one month number of transaction (first 10 customers)

bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df["Frequency"],
                                                        cltv_df["Recency"],
                                                        cltv_df["T"]).sort_values(ascending = False).head(10)

# Adding a new variable that predicts number of transaction by using BG/NBD model (expected one month sales)

cltv_df["expected_one_month_sales"] = bgf.predict(4,
                                                    cltv_df["Frequency"],
                                                    cltv_df["Recency"],
                                                    cltv_df["T"])

cltv_df.head()

# Plotting the expected number of transaction by transaction and customers

plot_period_transactions(bgf)
plt.show()

##################################################################################################################
# Gamma-Gamma Model(Finding Expected Average Profit)
##################################################################################################################

ggm = GammaGammaFitter(penalizer_coef=0.001)
ggm.fit(cltv_df["Frequency"], cltv_df["Monetary"])

ggm.conditional_expected_average_profit(cltv_df["Frequency"],
                                        cltv_df["Monetary"]).sort_values(ascending = False).head(10)

##################################################################################################################
# Creating Customer Life Time Value Prediction Model (for three months)
##################################################################################################################

cltv = ggm.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   freq = "W",
                                   time = 3,
                                   discount_rate = 0.01)

cltv.head()

cltv_final = cltv_df.merge(cltv, on = "CUSTOMERNAME", how = "left")
cltv_final = cltv_final.reset_index()
cltv_final.head()

scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(cltv_final[["clv"]])
cltv_final["CLTV"] = scaler.transform(cltv_final[["clv"]])
cltv_final.head()

cltv_final.sort_values(by = "CLTV", ascending = False).head()

##################################################################################################################
# Creating CLTV Segments
##################################################################################################################

## Creating Segments According to the Quantiles

cltv_final["Quantile_Segment"] = pd.qcut(cltv_final["clv"], 4, labels = ["D", "B", "C", "A"])

cltv_final.groupby("Quantile_Segment").agg(["mean", "count"])

## Creating Segments According to the Pareto

q_80 = cltv_final["CLTV"].quantile(0.8)

cltv_final["Pareto_Segment"] = ["High" if i >= q_80 else "Low" for i in cltv_final["CLTV"]]

cltv_final.groupby("Pareto_Segment").agg(["mean", "count"])




