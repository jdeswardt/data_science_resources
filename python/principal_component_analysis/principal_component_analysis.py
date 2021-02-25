################################################################################

#GJ DE SWARDT
#PRINCIPAL COMPONENT ANALYSIS (PCA)

################################################################################
#1.) SETUP

import pandas
import numpy
import plotly.express
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

################################################################################
#2.) DATA

##Import csv data
df_us_arrests = pandas.read_csv(filepath_or_buffer='~/Documents/GitHub/data_science_resources/data/us_arrests.csv',
                                names=['state',
                                       'murder',
                                       'assualt',
                                       'population',
                                       'rape'])

##Check dataframe
print(df_us_arrests)

##Create dataframe with only the independent X variables
df_features = df_us_arrests[['murder',
                             'assualt',
                             'population',
                             'rape']]

##Check dataframe
print(df_features)

################################################################################
#2.) EXPLORATORY DATA ANALYSIS

##Check summary statistics
df_features.describe()

##Check pairplots
pair_plots = plotly.express.scatter_matrix(data_frame=df_features)
pair_plots.show()

################################################################################
#3.) IMPLEMENT PCA

##3.1) Standardize features
features_standardized = StandardScaler().fit_transform(df_features)
df_features_standardized = pandas.DataFrame(data=features_standardized,
                                            columns=['murder',
                                                     'assualt',
                                                     'population',
                                                     'rape'])

##Check dataframe
print(df_features_standardized)

##3.2) Create principal components for each feature
##Specify number of components, create p components for p features
pca = PCA(n_components=4)

##Create components on standardized features
principal_components = pca.fit_transform(df_features_standardized)

##Create dataframe with principal components
df_prin_comp = pandas.DataFrame(data=principal_components,
                                columns=['principal_component_1',
                                         'principal_component_2',
                                         'principal_component_3',
                                         'principal_component_4'])

##Check dataframe
print(df_prin_comp)

##3.3) Evaluate number of components required
##Number of principal components
pc_values = numpy.arange(pca.n_components_) + 1

##Variation explained by each component
variation_explained = pca.explained_variance_ratio_

##Create scree plot
fig = plotly.express.line(x=pc_values,
                          y=variation_explained,
                          title='Scree plot',
                          labels={'x': 'Number of principal components',
                                  'y': 'Proportion of variance explained'})
fig.update_traces(mode='markers+lines')
fig.show()

##Calculate the cumulative variance explained
cumul_var_explained = numpy.cumsum(pca.explained_variance_ratio_)

##Create a area plot
fig = plotly.express.area(x=range(1, cumul_var_explained.shape[0] + 1),
                          y=cumul_var_explained,
                          labels={'x': 'Number of principal components',
                                  'y': 'Cumulative variance explained'})
fig.update_traces(mode='markers+lines')
fig.show()

################################################################################
#4.) PLOTLY EXPRESS BIPLOT

##Create feature names list
features = ['murder',
            'assualt',
            'population',
            'rape']

##Create dataframe showing the state
state = df_us_arrests['state']

##Calculate loadings
loadings = pca.components_.T * numpy.sqrt(pca.explained_variance_)

##Create scatter object
biplot = plotly.express.scatter(data_frame=df_prin_comp,
                                x='principal_component_1',
                                y='principal_component_2',
                                text=state,
                                title='Biplot for US arrests data',
                                labels={'principal_component_1': 'PC1',
                                        'principal_component_2': 'PC2'})

##Adjust data labels
biplot.update_traces(textposition='top center')

##Add vectors to plot
for i, feature in enumerate(features):
    biplot.add_shape(type='line',
                     x0=0,
                     y0=0,
                     x1=loadings[i, 0],
                     y1=loadings[i, 1])
    biplot.add_annotation(x=loadings[i, 0],
                          y=loadings[i, 1],
                          ax=0,
                          ay=0,
                          xanchor='center',
                          yanchor='bottom',
                          text=feature)

##Show plot
biplot.show()

################################################################################
