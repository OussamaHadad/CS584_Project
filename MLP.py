import tensorflow as tf
from tensorflow import keras

class MLP(keras.Model):
    def __init__(self, num_users: int, num_items: int, factors= 8, optimizer= tf.optimizers.Adam(), loss= "binary_crossentropy", num_layers= 3, **kwargs):
        super().__init__(**kwargs)
        userId_input = keras.Input(shape=(1,), name='userID')
        itemId_input = keras.Input(shape=(1,), name='itemID')

        # Compute the embeddings of userId and itemId
        embedding_size = 2*factors
        userId_embedding = keras.layers.Embedding(input_dim= num_users, output_dim= embedding_size, input_length= 1, name= 'user_embedding')(userId_input)
        itemId_embedding = keras.layers.Embedding(input_dim= num_items, output_dim= embedding_size, input_length= 1, name= 'item_embedding')(itemId_input)

        # Flatten the embedding vectors
        userId_embedding = keras.layers.Flatten()(userId_embedding)
        itemId_embedding = keras.layers.Flatten()(itemId_embedding)

        output_MLP = keras.layers.Concatenate()([userId_embedding, itemId_embedding])
        for i in range(num_layers-1, -1, -1):
            output_MLP = keras.layers.Dense(units=(2**i)*factors, activation='relu')(output_MLP)    # Size of hidden layer number i from the end is 8*(2**i) with i>= 0
        output_MLP = keras.layers.Dense(units=1, activation='sigmoid', name='prediction_MLP')(output_MLP)

        self.model = keras.Model(inputs = [userId_input, itemId_input], outputs= output_MLP)
        self.model.compile(optimizer= optimizer, loss= loss)