import tensorflow as tf
from tensorflow import keras

class MF(keras.Model):
    def __init__(self, num_users: int, num_items: int, factors= 8, optimizer= tf.optimizers.Adam(), loss= "binary_crossentropy", **kwargs):
        super().__init__(**kwargs)
        userId_input = keras.Input(shape=(1,), name='userID')
        itemId_input = keras.Input(shape=(1,), name='itemID')

        # Compute the embeddings of userId and itemId
        embedding_size= 2*factors
        userId_embedding = keras.layers.Embedding(input_dim= num_users, output_dim= embedding_size, input_length= 1, name= 'user_embedding')(userId_input)
        itemId_embedding = keras.layers.Embedding(input_dim= num_items, output_dim= embedding_size, input_length= 1, name= 'item_embedding')(itemId_input)

        # Flatten the embedding vectors
        userId_embedding = keras.layers.Flatten()(userId_embedding)
        itemId_embedding = keras.layers.Flatten()(itemId_embedding)

        # axes=(1, 1) -> multiply on the second axis of both vectors
        output_MF = keras.layers.Dot(axes=(1, 1))([userId_embedding, itemId_embedding])

        self.model = keras.Model(inputs = [userId_input, itemId_input], outputs= output_MF)
        self.model.compile(optimizer= optimizer, loss= loss)