import pandas as pd
data=pd.read_csv("netflix_titles.csv")
#Copy path from content root

print(data.head())
print(data.shape)

missing_values_count = data.isnull().sum()
print(missing_values_count[0:1])

"""
#new_data = data.dropna()
#new_data = data.dropna(axis = 1) #drops columns
#print(new_data.shape)

#cleaned_data = data.fillna(0)
cleaned_data = data.fillna(method='bfill', axis=0).fillna(0)
missing_values_count = data.isnull().sum()
print(missing_values_count)
"""