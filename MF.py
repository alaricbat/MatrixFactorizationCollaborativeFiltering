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
        self.total_of_ratings = Y_data.shape[0]
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
        self.Y_normalized = self.Y_data.copy()

    def __normalize_Y(self):

        if self.user_based == True:
            user_cols = 0
            item_cols = 1
            n_objects = self.n
        else: # item-based
            user_cols = 1
            item_cols = 0
            n_objects = self.m
        
        users = self.Y_data[:, user_cols]
        self.mu = np.zeros((n_objects,))
        
        for i in range (n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            user_ids = np.where(users == i)[0].astype(np.int32)
            # indicates of all ratings associated with user n
            item_ids = self.Y_data[user_ids, item_cols]
            # corresponding ratings
            ratings = self.Y_data[user_ids, 2]
            mean_rating = np.mean(ratings)
            if np.isnan(mean_rating):
                m = 0 # to avoid empty array and nan value
            self.mu[i] = mean_rating
            # normalize
            self.Y_data[user_ids, 2] = ratings - self.mu[i]

            pass


    def loss_function(self):
        """
        formula = r"L(W) = (1/(2*s)) * Σ_{n=1}^{N} Σ_{m: r_{mn}=1} (y_{mn} - x_m w_n)^2 + (λ/2) * ||W||_F^2"
        """
        loss_value = 0
        for i in range(self.total_of_ratings):
            # user
            user_id = int(self.Y_normalized[i, 0])
            item_id = int(self.Y_normalized[i, 1])
            rate_number = int(self.Y_normalized[i, 2])
            loss_value += (rate_number - np.dot(self.X[item_id, :], self.W[:, user_id]))**2
        
        loss_value /= (2 * self.total_of_ratings) # 1/(2*s)

        #Chosen norm: 'fro'  Frobenius norm  
        loss_value += self.lamb * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))

        return loss_value

    def get_items_rated_by_user(self, user_id):
        users_idx = np.where(self.Y_data[:, 0] == user_id)[0]
        item_ids = self.Y_data[users_idx, 1].astype(np.int32)
        ratings = self.Y_data[users_idx, 2].astype(np.int32)
        return (item_ids, ratings)
    
    def get_users_who_rate_item(self, item_id):
        item_idx = np.where(self.Y_data[:, 1] == item_id)[0]
        user_ids = self.Y_data[item_idx, 0].astype(np.int32)
        ratings = self.Y_data[item_idx, 2].as_type(np.int32)
        return (user_ids, ratings)
    

    def updateX(self):
        """
        x_m = x_m - η * ( - (1/s) * (y_m - x_m * W_m) * W_m^T + λ * x_m )
        """
        for item in range(self.m):
            user_ids, ratings = self.get_users_who_rate_item(item)
            W_m = self.W[:, user_ids] # size (K, n*)
            X_m = self.X[item, :] # size (1, K)
            # ratings = size (n*, 1)
            grad = (np.dot((ratings.T - np.dot(X_m, W_m)), W_m.T) + (self.lamb * X_m)) * (-(1/self.total_of_ratings)) 
            X_m -= self.learning_rate * grad.reshape((self.K,))
            self.X[item, :] = X_m
            #self.X[m, :] -= self.learning_rate* => grad_xm.reshape((self.K,))

    def updateW(self):
        """
        w_n = w_n - η * ( - (1/s) * X_n^T * (y_n - X_n * w_n) + λ * w_n )
        """
        for user in range (self.n):
            item_ids, ratings = self.get_items_rated_by_user(user)
            W_n = self.W[:, user] # size (K, 1)
            X_n = self.X[item_ids, :] # size (m*, K)
            # ratings = size (m*, 1)
            grad = (np.dot((ratings - np.dot(X_n, W_n)).T, X_n).T + (self.lamb * W_n)) * (-(1/self.total_of_ratings))
            W_n -= self.learning_rate * grad.reshape((self.K,))
            self.W[:, user] = W_n

    def fit(self):
        self.__normalize_Y()
        for iter in range(self.max_iter):
            self.updateW()
            self.updateX()

    

            
    



        

    

