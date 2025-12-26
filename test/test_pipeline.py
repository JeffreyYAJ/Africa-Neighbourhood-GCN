import unittest
import numpy as np
import scipy.sparse as sp
import torch
from src.utils import normalize_adjacency

class TestGCNUtils(unittest.TestCase):
    
    def test_normalization_mechanics(self):
        
        print("\n TEST:")

        raw_adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        adj_sparse = sp.coo_matrix(raw_adj)
        
        result = normalize_adjacency(adj_sparse)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 3))
        
        diag_sum = torch.trace(result)
        self.assertGreater(diag_sum, 0, "La diagonale ne devrait pas être vide (Self-loops manquants?)")
        
        print("Utils Normalization passed!")

if __name__ == '__main__':
    unittest.main()