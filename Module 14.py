# ---------------------------------------------------------
# Import plotting class from matplotlib library
from matplotlib import pyplot as plt

# Create a scatter plot
plt.scatter(x, y)

# Display the scatter plot
plt.show()

# ---------------------------------------------------------
# Import linkage and fcluster functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()

# ---------------------------------------------------------
# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Compute cluster centers
centroids,_ = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()

# ---------------------------------------------------------
# Import the whiten function
from scipy.cluster.vq import whiten

goals_for = [4,3,2,3,1,1,2,0,1,4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)

# ---------------------------------------------------------
# Plot original data
plt.plot(goals_for, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

# Show the legend in the plot
plt.legend()

# Display the plot
plt.show()

# ---------------------------------------------------------
# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function to standardize the data
scaled_data = whiten(rate_cuts)

# Plot original data
plt.plot(rate_cuts, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

plt.legend()
plt.show()

# ---------------------------------------------------------
# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind='scatter')
plt.show()

# ---------------------------------------------------------
# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind = 'scatter')
plt.show()

# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())

# ---------------------------------------------------------
# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import fcluster, linkage

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data = comic_con)
plt.show()

# ---------------------------------------------------------
# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'single', metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data = comic_con)
plt.show()

# ---------------------------------------------------------
# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='complete', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled',
                hue='cluster_labels', data = comic_con)
plt.show()

# ---------------------------------------------------------
# Import the pyplot class
from matplotlib import pyplot as plt

# Define a colors dictionary for clusters
colors = {1:'red', 2:'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x='x_scaled',
                	   y='y_scaled',
                	   c=comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.show()

# ---------------------------------------------------------
# Import the seaborn module
import seaborn as sns

# Plot a scatter plot using seaborn
sns.scatterplot(x='x_scaled',
                y='y_scaled',
                hue='cluster_labels',
                data = comic_con)
plt.show()

# ---------------------------------------------------------
# Import the dendrogram function
from scipy.cluster.hierarchy import dendrogram

# Create a dendrogram
dn = dendrogram(distance_matrix)

# Display the dendogram
plt.show()

# ---------------------------------------------------------
%timeit linkage(comic_con[['x_scaled', 'y_scaled']], method='ward', metric='euclidean')

# ---------------------------------------------------------
# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', hue='cluster_labels', data=fifa)
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


