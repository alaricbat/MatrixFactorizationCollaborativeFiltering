import numpy as np

class MF(object):

    def __init__(self,
                 Y_data,
                 K,
                 lamb=0.1,
                 Xinit=None,
                 Winit=None,
                 learning_rate=0.5,
                 max_iter=1000,
                 user_based=True):
        # Y has size (MxN) => N is number of users and M is number of items 
        self.Y_data = Y_data
        self.K = K
        # regularization parameter
        self.lamb = lamb
        # learning rate for gradient descent
        self.learning_rate = learning_rate
        # maximum number of interations
        self.max_iter = max_iter
        # determine user-based or item-based
        self.user_based = user_based
        # number of users, items and ratings. Add more one since id starts from 0
        self.n = int(np.max(Y_data[:, 0])) + 1
        self.m = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]
        # X has size (KxM) => K is optional number and smaller than M
        # X is Item features
        if Xinit == None:
            #random.randn() is a function that generates random numbers from a normal distribution with Mean = 0 and Standard Deviation = 1
            #creating a two-dimensional matrix with dimensions m rows × K columns
            self.X = np.random.randn(self.m, K)
        else: 
            self.X = Xinit
        # W has size (KxN) => K is optional number and smaller than N
        # W is user features
        if Winit == None:
            #random.randn() is a function that generates random numbers from a normal distribution with Mean = 0 and Standard Deviation = 1
            #creating a two-dimensional matrix with dimensions K rows × n columns
            self.W = np.random.randn(K, self.n)
        else:
            self.W = Winit
        # normalized data, update later in normaized_Y function
        self.Y_data_n = self.Y_data.copy()

    def normalize_Y(self):

        if self.user_based == True:
            user_cols = 0
            item_cols = 1
            n_objects = self.n
        else: # item-based
            user_cols = 1
            item_cols = 0
            n_objects = self.m
        

    

