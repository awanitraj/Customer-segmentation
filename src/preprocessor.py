from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the data for clustering by scaling the relevant columns.
    """
 
    features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    
    return scaled_data
