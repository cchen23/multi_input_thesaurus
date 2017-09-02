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
    # TODO: profile and try to speed up.
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
    print("Starting Words: %s" % (", ".join(starting_words)))
    print("Closest %d neighbors (can include starting words): " % (num_closest + 1))
    for index in closest_indices:
        neighbor_word = words[index]
        print(neighbor_word)
        if neighbor_word not in starting_words:
            neighbors_words.append(neighbor_word)
    return neighbors_words

if __name__ == '__main__':
    run_recommender = True
    embeddings_filename_50 = 'glove/glove.6B.50d.txt'
    embeddings_filename_100 = 'glove/glove.6B.100d.txt'
    embeddings_filename_200 = 'glove/glove.6B.200d.txt'
    embeddings_filename_300 = 'glove/glove.6B.300d.txt'

    # Reading embeddings takes awhile, so don't read until necessary.
    embeddings_df_50 = pd.DataFrame()
    embeddings_df_100 = pd.DataFrame()
    embeddings_df_200 = pd.DataFrame()
    embeddings_df_300 = pd.DataFrame()

    while (run_recommender):
        embeddings_length = input("What dimension word vectors would you like to use? (50, 100, 200, or 300)\n")
        while embeddings_length not in ["50", "100", "200", "300"]:
            embeddings_length = input("Word vector dimension options are 50, 100, 200, or 300. Which dimension would you like to use?\n")
        if embeddings_length == "50":
            if embeddings_df_50.empty:
                embeddings_df_50 = read_embeddings(embeddings_filename_50)
            embeddings_df = embeddings_df_50
        elif embeddings_length == "100":
            if embeddings_df_100.empty:
                embeddings_df_100 = read_embeddings(embeddings_filename_100)
            embeddings_df = embeddings_df_100
        elif embeddings_length == "200":
            if embeddings_df_200.empty:
                embeddings_df_200 = read_embeddings(embeddings_filename_200)
            embeddings_df = embeddings_df_200
        else:
            if embeddings_df_300.empty:
                embeddings_df_300 = read_embeddings(embeddings_filename_300)
            embeddings_df = embeddings_df_300

        num_closest = input("How many synonyms would you like to receive?\n")
        while not num_closest.isdigit():
            num_closest = input("How many synonyms would you like to receive? Please enter a non-negative integer.\n")
        num_closest = int(num_closest)

        starting_words = input("Enter your list of starting words, separated by commas.\n")
        starting_words = starting_words.split(",")
        starting_words = [word.strip() for word in starting_words]

        recommended_words = find_closest_words(starting_words, num_closest, embeddings_df)

        run_recommender = input("Would you like to generate more recommendations? Y/N\n")
        while run_recommender not in ["Y", "y", "N", "n"]:
            run_recommender = input("Would you like to generate more recommendations? Please enter Y/N\n")
        run_recommender = run_recommender in ["Y", "y"]
