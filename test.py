import pandas as pd
data=pd.read_csv("country_vaccinations.csv")
#Copy path from content root

print(data.head())
print(data.shape)