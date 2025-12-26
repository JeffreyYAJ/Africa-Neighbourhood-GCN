import torch
import numpy as np
import scipy.sparse as sp

def load_africa_mockdata():
    countries = [
        "Nigeria", "Egypt", "SouthAfrica", "Algeria", "Morocco", 
        "Kenya", "Ethiopia", "Ghana", "IvoryCoast", "Cameroon", 
        "Niger", "Chad", "Sudan", "Benin", "Togo"
    ]
    
    country_to_index = {name: i for i, name in enumerate(countries)}
    
    # Features: [GDP ($), Population (M)] 
    raw_features = np.array([
        [440.8, 206.1], [363.1, 102.3], [301.9, 59.3], [145.0, 43.8], [112.8, 36.9],
        [98.8,  53.7],  [107.6, 115.0], [72.4,  31.0], [61.3,  26.4], [39.8,  26.5],
        [12.9,  24.2],  [11.7,  16.4],  [21.2,  43.8],  [15.6,  12.1], [7.5,   8.3]
    ], dtype=np.float32)
    
    features = (raw_features - raw_features.mean(axis=0)) / raw_features.std(axis=0)
    features = torch.FloatTensor(features)

    # Labels: 0: West, 1: North, 2: East/South/Central
    labels = torch.LongTensor([0, 1, 2, 1, 1, 2, 2, 0, 0, 2, 0, 2, 1, 0, 0])

    # Neoghbourhood adjencey matrix based on country borders
    border_pairs = [
        ("Nigeria", "Benin"), ("Nigeria", "Niger"), ("Nigeria", "Chad"), ("Nigeria", "Cameroon"),
        ("Benin", "Togo"), ("Benin", "Niger"), ("Ghana", "Togo"), ("Ghana", "IvoryCoast"),
        ("Niger", "Algeria"), ("Niger", "Chad"), ("Niger", "Benin"),
        ("Chad", "Sudan"), ("Chad", "Cameroon"), ("Chad", "Niger"),
        ("Sudan", "Egypt"), ("Sudan", "Ethiopia"), ("Sudan", "Chad"),
        ("Algeria", "Morocco"), ("Algeria", "Niger"),
        ("Cameroon", "Nigeria"), ("Cameroon", "Chad"),
        ("Ethiopia", "Sudan"), ("Ethiopia", "Kenya"),
        ("SouthAfrica", "Nigeria")
    ]
    
    adj_mat = np.zeros((len(countries), len(countries)))
    for country1, country2 in border_pairs:
        if country1 in country_to_index and country2 in country_to_index:
            adj_mat[country_to_index[country1]][country_to_index[country2]] = 1
            adj_mat[country_to_index[country2]][country_to_index[country1]] = 1 

    idx_train = torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13])
    idx_test  = torch.LongTensor([5, 12, 14]) 

    return sp.coo_matrix(adj_mat), features, labels, countries, idx_train, idx_test