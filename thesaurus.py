import numpy as np
import pandas as pd

""" Creates a dataframe containing the embeddings of each word, with the words
set as the index of the dataframe. """
def read_embeddings(filename):
    word_embeddings = pd.read_table(filename, header=None, sep=" ", index_col=0, quoting=3)
    return word_embeddings

""" Prints the num_closest closest words to the words in starting_words based
on cosine similarity of word embeddings. Returns a list of all the words
(which can include the starting words). """
def find_closest_words(starting_words, num_closest, embeddings_df):
    embeddings_matrix = embeddings_df.as_matrix()
    embedding_vectors_norms = np.linalg.norm(embeddings_matrix, axis=1) # nx1 matrix, where row i contains norm of vector i
    starting_words_vector = np.zeros(embeddings_matrix[0].size)
    for starting_word in starting_words:
        # TODO: check if embeddings_df has the starting word. Print a warning if it does not.
        starting_words_vector += embeddings_df.loc[starting_word]
    starting_words_vector /= len(starting_words)

    neighbors_words = []
    dot_prods = np.dot(embeddings_matrix, starting_words_vector) # nx1 matrix, where row i contains dot(word_i_vec, seed_word_vec)
    cosine_similarities = dot_prods / embedding_vectors_norms # just want maximum cosine similarity, so don't worry about dividing by norm of starting word (same for each one)
    closest_indices = cosine_similarities.argsort()[-(num_closest+1):][::-1]
    words = embeddings_df.index.values
    print("Starting Words: %s" % (starting_words)) # TODO: join into prettier string.
    print("Closest %d neighbors (can include starting words): " % (num_closest + 1))
    for index in closest_indices:
        neighbor_word = words[index]
        print(neighbor_word)
        if neighbor_word not in starting_words:
            neighbors_words.append(neighbor_word)
    return neighbors_words

if __name__ == '__main__':
    embeddings_filename = 'glove/glove.6B.50d.txt'
    embeddings_df = read_embeddings(embeddings_filename)
    starting_words = ["happy", "joyful", "content"]
    recommended_words = find_closest_words(starting_words, 10, embeddings_df)
    starting_words = ["happy", "excited", "nervous"]
    recommended_words = find_closest_words(starting_words, 10, embeddings_df)
    starting_words = ["down", "sad"]
    recommended_words = find_closest_words(starting_words, 10, embeddings_df)
    starting_words = ["down", "up"]
    recommended_words = find_closest_words(starting_words, 10, embeddings_df)