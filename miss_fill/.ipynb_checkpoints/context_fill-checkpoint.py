'''
All the missing value filling methods that's call for the context
'''
## libraries
import copy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore


class ContextFill:
    
    def __init__(self, outcome_name = 'outcome'):
        
        self.outcome_name = outcome_name
        


    def get_context_vectors(self, without_miss_dt, context_cols):
        
        context_vectors = []
        for row in without_miss_dt.itertuples():
            row_values = [getattr(row, col) for col in context_cols]
            context_vectors.append(row_values)

        return (np.array(context_vectors))

    def get_distance(self, context_vectors, miss_vec):
        
        distances = []
        for vector in context_vectors:
            d = np.linalg.norm(vector - miss_vec)
            distances.append(d) 
        return(distances)



    def KNN_fill(self, dt = pd.DataFrame, context_cols = list, k = 1): 
        # dt is the data Frame
        # context_cols will be list of columns name which are considered as context 
        
        dt_fill = dt.copy()
        miss_dt = dt_fill[dt_fill['outcome'].isna()]
        without_miss_dt = dt_fill[dt_fill['outcome'].isna() == False].copy()
        context_vectors = self.get_context_vectors(without_miss_dt, context_cols)
        
        for row in miss_dt.itertuples():
            miss_vec = np.array([getattr(row, col) for col in context_cols])
            dis = self.get_distance(context_vectors, miss_vec)
            without_miss_dt['distance'] = dis
            
            sorted_without_miss_dt = without_miss_dt.sort_values(by = 'distance')
            
            fill_value = sorted_without_miss_dt['outcome'].head(k).mean()
            dt_fill.loc[row.Index,'outcome'] = fill_value
        
        return dt_fill

    def cluster_fill(self, dt, context_cols = list, N = 2, m = 1): 
        # dt is the data Frame
        # context_cols will be list of columns name which are considered as context 
        # N represents the number of clusters for KMean
        
        dt_fill = dt.copy()
        miss_dt = dt_fill[dt_fill['outcome'].isna()]
        without_miss_dt = dt_fill[dt_fill['outcome'].isna() == False].copy()
        context_dt = without_miss_dt[context_cols]
        
        ## apply KMeans clustering to context_dt
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(context_dt)

        kmeans = KMeans(n_clusters= N, random_state=0)  # Choose the number of clusters (2 in this case)
        kmeans.fit(scaled_data)

        without_miss_dt['cluster'] = kmeans.labels_  # Add cluster labels to the dataframe
        centroids = kmeans.cluster_centers_  # Get cluster centroids
        
        
        
        
        for row in miss_dt.itertuples():
            miss_vec = np.array([getattr(row, col) for col in context_cols])
            
            ## calculate the distance from the clustres cenroids
            dis = self.get_distance(centroids, miss_vec)
            dis_dt = pd.Series(dis)
                    
            sorted_dis_dt = dis_dt.sort_values()
            selected_cluster = sorted_dis_dt.index[:m].values ## to get m nearest clusters
            
            ## calculate the missing value with above mentioned equation (1)
            clusters_mean = without_miss_dt.groupby('cluster')['outcome'].mean()
            inverse_dis = 1 / dis_dt.loc[selected_cluster]
            numerator = (clusters_mean.loc[selected_cluster] / dis_dt.loc[selected_cluster]).mean()
            denominator = inverse_dis.mean()
            
            fill_value = numerator/denominator 
            
            ## fill the missing value in appropiate place 
            dt_fill.loc[row.Index,'outcome'] = fill_value
        
        return dt_fill

    def fit_estimate_beta(self,df, context_cols, lamb): ## lamb is the regulization parameter
        ## prepare for estimator calculate beta
        lambI = np.eye(len(context_cols)+ 1)*lamb
        X = df[context_cols].values
        y = df['outcome'].values
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        beta_DR = np.linalg.inv(X_b.T.dot(X_b) + lambI ).dot(X_b.T).dot(y)
        
        return beta_DR

    def predict(self,beta_DR, X):
        """
        Predict target values for given input features X.
        X: numpy array of shape (n_samples, n_features)
        """
        return beta_DR[0] + X.dot(beta_DR[1:])

    def DR_fill(self, dt, lamb, context_cols = list): 
        # dt is the data Frame
        # context_cols will be list of columns name which are considered as context 
        
        dt_fill = dt.copy()
        miss_dt = dt_fill[dt_fill['outcome'].isna()]
        without_miss_dt = dt_fill[dt_fill['outcome'].isna() == False].copy()
        beta_DR = self.fit_estimate_beta(without_miss_dt, context_cols,lamb)
        
        for row in miss_dt.itertuples():
            miss_vec = np.array([getattr(row, col) for col in context_cols])
            
            fill_value = self.predict(beta_DR, miss_vec)
            dt_fill.loc[row.Index,'outcome'] = fill_value
        
        return dt_fill