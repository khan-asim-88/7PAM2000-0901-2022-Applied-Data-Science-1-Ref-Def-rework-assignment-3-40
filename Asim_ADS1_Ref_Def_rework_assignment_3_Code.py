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
