# ---------------------------------------------------------
# Display the number of rows and columns
nsfg.shape

# ---------------------------------------------------------
# Display the number of rows and columns
nsfg.shape

# Display the names of the columns
nsfg.columns

# ---------------------------------------------------------
# Display the number of rows and columns
nsfg.shape

# Display the names of the columns
nsfg.columns

# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']

# ---------------------------------------------------------
# Display the number of rows and columns
nsfg.shape

# Display the names of the columns
nsfg.columns

# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']

# Print the first 5 elements of ounces
print(ounces.head())

# ---------------------------------------------------------
# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace(8, np.nan, inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())

# ---------------------------------------------------------
# Select the columns and divide by 100
agecon = nsfg['agecon']/100
agepreg = nsfg['agepreg']/100

# ---------------------------------------------------------
# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# ---------------------------------------------------------
# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())

# ---------------------------------------------------------
# Plot the histogram
plt.hist(agecon, bins=20, ____='____')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

# ---------------------------------------------------------
# Plot the histogram
plt.hist(agecon, bins=20, histtype='step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

# ---------------------------------------------------------
# Create a Boolean Series for full-term babies
full_term = nsfg['prglngth'] >= 37

# Select the weights of full-term babies
full_term_weight = birth_weight[full_term]

# Compute the mean weight of full-term babies
print(full_term_weight.mean())

# ---------------------------------------------------------
# Filter full-term babies
full_term = nsfg['prglngth'] >= 37

# Filter single births
single = nsfg['nbrnaliv'] == 1

# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[full_term & single]
print('Single full-term mean:', single_full_term_weight.mean())

# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[full_term & ~single]
print('Multiple full-term mean:', mult_full_term_weight.mean())


# ---------------------------------------------------------
# Compute the PMF for year
pmf_year = Pmf(gss['year'], normalize=False)

# Print the result
print(pmf_year)

# ---------------------------------------------------------
# Select the age column
age = gss['age']


# ---------------------------------------------------------
# Select the age column
age = gss['age']

# Make a PMF of age
pmf_age = Pmf(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()

# ---------------------------------------------------------
# Select the age column
age = gss['age']

# Compute the CDF of age
cdf_age = Cdf(age)

# Calculate the CDF of 30
print(cdf_age(30))

# ---------------------------------------------------------
# Calculate the 75th percentile
percentile_75th = cdf_income.inverse(0.75)

# Calculate the 25th percentile
percentile_25th = cdf_income.inverse(0.25)

# Calculate the interquartile range
iqr = percentile_75th - percentile_25th

# Print the interquartile range
print(iqr)

# ---------------------------------------------------------
# Select realinc
income = gss['realinc']

# Make the CDF
cdf_income = Cdf(income)

# Plot it
cdf_income.plot()

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.show()

# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


# ---------------------------------------------------------


