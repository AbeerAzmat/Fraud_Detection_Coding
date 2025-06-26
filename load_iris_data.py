import seaborn as sns

iris_df = sns.load_dataset('iris')
print(iris_df.head())

# Split features and labels
X = iris_df.drop('species', axis=1).values
y = iris_df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values
