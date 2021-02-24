################################################################################

#GJ DE SWARDT
#PRINCIPAL COMPONENT ANALYSIS (PCA)

################################################################################
#1.) SETUP

import pandas
import plotly.express
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
#df_us_arrests = pandas.read_csv(, names=['sepal length','sepal width','petal length','petal width','target'])

################################################################################
#2.) DATA

##Import csv data
df_us_arrests = pandas.read_csv(filepath_or_buffer='~/Documents/GitHub/data_science_resources/data/us_arrests.csv',
                                names=['state',
                                       'murder',
                                       'assualt',
                                       'population',
                                       'rape'])

##Create dataframe with only the independent X variables
df_features = df_us_arrests[['murder',
                             'assualt',
                             'population',
                             'rape']]

################################################################################
#2.) EXPLORATORY DATA ANALYSIS

##Check summary statistics
df_features.describe()

##Check pairplots
pair_plots = plotly.express.scatter_matrix(data_frame=df_features)
pair_plots.show()

##Standardize features
features_standardized = StandardScaler().fit_transform(df_features)
df_features_standardized = pandas.DataFrame(data=features_standardized,
                                            columns=['murder',
                                                     'assualt',
                                                     'population',
                                                     'rape'])
print(df_features_standardized)

################################################################################
#X.) PCA BENEFITS FOR VISUALIZATION

##Add pairplots

################################################################################
#X.) IMPLEMENT PCA

##Specify number of components
pca = PCA(n_components=2)

##Implement pca on standardized features
principalComponents = pca.fit_transform(df_features_standardized)



principalDf = pandas.DataFrame(data=principalComponents,
                               columns=['principal_component_1',
                                        'principal_component_2'])









################################################################################
