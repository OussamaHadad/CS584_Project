import tensorflow as tf
from tensorflow import keras

class NeuMF(keras.Model):
    def __init__(self, num_users: int, num_items: int, factors= 8, optimizer= tf.optimizers.Adam(), loss= "binary_crossentropy", num_layers= 3, **kwargs):
        super().__init__(**kwargs)
        userId_input = keras.Input(shape=(1,), name='userID')
        itemId_input = keras.Input(shape=(1,), name='itemID')
        # MLP part
        MLP_embedding_size = GMF_embedding_size= 2*factors
        user_embedding_MLP = keras.layers.Embedding(input_dim= num_users, output_dim= MLP_embedding_size, name='user_embedding_MLP')(userId_input)
        item_embedding_MLP = keras.layers.Embedding(input_dim= num_items, output_dim= MLP_embedding_size, name='item_embedding_MLP')(itemId_input)
        user_latent_MLP = keras.layers.Flatten()(user_embedding_MLP)
        item_latent_MLP = keras.layers.Flatten()(item_embedding_MLP)
        concatenation_embeddings = keras.layers.Concatenate()([user_latent_MLP, item_latent_MLP])
        output_MLP = concatenation_embeddings
        layer_name = 0
        for i in range(num_layers-2,-2,-1):
                # Size of hidden layer number i from the end is 8*(2**i) with i>= 0
                layer = keras.layers.Dense(units=(2**i)*factors, activation='relu', name='layer%d' %(layer_name+1))
                output_MLP = layer(output_MLP)
                layer_name += 1
        # GMF part
        user_embedding_GMF = keras.layers.Embedding(input_dim= num_users, output_dim= GMF_embedding_size, name='user_embedding_GMF')
        item_embedding_GMF = keras.layers.Embedding(input_dim= num_items, output_dim= GMF_embedding_size, name='item_embedding_GMF')
        user_latent_GMF = keras.layers.Flatten()(user_embedding_GMF(userId_input))
        item_latent_GMF = keras.layers.Flatten()(item_embedding_GMF(itemId_input))
        mul = keras.layers.Multiply()([user_latent_GMF, item_latent_GMF])
        # NeuMF  
        concatenation_of_models = keras.layers.Concatenate(name='final_concatenation')([mul, output_MLP]) # len = factors
        prediction_NeuMF = keras.layers.Dense(units=1, activation='sigmoid', name='prediction')(concatenation_of_models)

        self.model = keras.Model(inputs = [userId_input, itemId_input], outputs= prediction_NeuMF)
        self.model.compile(optimizer= optimizer, loss= loss)