#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycountry
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from err_ranges import err_ranges
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


# In[2]:


# Define a function that reads World Bank data from a CSV file and returns two dataframes:
def read_worldbank_data(filename):
    """
    Function is used to load the data and then it convert it into two formats
    One is year as columns and other is country name as column.
    
    Parameter:
        filename: Name of the data file.
    
    Returns:
        years: Dataframe with years as column
        countries: Dataframe with country names as column
        
    """
    
    # Read the CSV file and skip the first 4 rows
    df = pd.read_csv(filename, skiprows=4)
    
    # Drop unnecessary columns
    df = df.iloc[:, :-1]
    
    # Create a copy of the dataset with years as columns
    years = df.copy()
    
    # Create a dataset with countries as columns
    countries = df.set_index(["Country Name", "Indicator Name"])
    countries.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)
    
    # Transpose the countries dataframe
    countries = countries.T
    
    # Return the years and countries dataframes
    return years, countries


df, countries = read_worldbank_data("dataset.csv")


# In[3]:


# Function to filter the data
def filter_data(df, indicators):
    """
    Function is used to extract specific data from the dataframe
    
    Parameters:
        df: Total Dataframe
        indicators: The indicators for which data has to be extracted.
        
    Returns:
        filtered_df: Extracted data
    """
    
    # Filter the dataset for the required indicators
    filtered_df = df[df["Indicator Name"].isin(indicators)]
    
    # Extracting data for only countries
    country_names = [country.name for country in list(pycountry.countries)]
    filtered_df = filtered_df[filtered_df["Country Name"].isin(country_names)]
    
    return filtered_df


# In[4]:


indicators = [
    'Agricultural land (% of land area)',
    'Forest area (% of land area)',
    'CO2 emissions (metric tons per capita)',
    'Methane emissions (kt of CO2 equivalent)',
    'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)',
    'Total greenhouse gas emissions (kt of CO2 equivalent)',
    'Renewable electricity output (% of total electricity output)'
    'Renewable energy consumption (% of total final energy consumption)',
]

filtered_df = filter_data(df, indicators)


# In[5]:


# Handle missing values
filtered_df = filtered_df.fillna(method='ffill').fillna(method='bfill')


# In[6]:


# Pivot the dataframe
pivot_df = filtered_df.pivot_table(index='Country Name', columns='Indicator Name', values='2020')


# In[7]:


labels = ["Agricultural land", "Forest area", "CO2 emissions", "Methane emissions", "Nitrous oxide emissions", 
          "Total greenhouse gas emissions", "Renewable electricity output", "Renewable energy consumption"]


# In[8]:


# Correlation
corr = pivot_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.title('Correlation between environmental factors')
plt.show()


# In[9]:


# Ploting Highest Forest area Countries
temp = pivot_df[["Forest area (% of land area)"]].sort_values(by="Forest area (% of land area)", ascending=False).iloc[:30]
temp.plot(kind="bar", figsize=(12, 5))
plt.ylabel("Forest Area")
plt.title("Forest of area of Countries")
plt.show()


# ## Clustering

# In[10]:


# Normalize the dataset
scaler = StandardScaler()
normalized_data = scaler.fit_transform(pivot_df.values)


# In[11]:


# Choose the number of clusters
n_clusters = 3

# Perform clustering using KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(normalized_data)

# Add the cluster labels to the dataset
pivot_df["Cluster"] = labels
