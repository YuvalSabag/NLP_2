"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""

# you can use these packages (uncomment as needed)
import pickle
import pandas as pd
import numpy as np
import os, time, re, sys, random, math, collections, nltk


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []
    with open(fn, 'r', encoding='utf-8') as f:
        text = f.read().lower()  # read the file and convert to lower case

    text = re.split(r'\n', text)  # split the text into lines

    for sentence in text:
        clen_sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        clean_sentence = re.sub(r'\s+', ' ', clen_sentence).strip()

        if clean_sentence:
            sentences.append(clen_sentence)

    return sentences


def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    with open(fn, 'rb') as f:
        sg_model = pickle.load(f)

    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        self.word_count = {}  # dictionary of word counts
        counts = collections.Counter()
        for sentence in sentences:
            counts.update(sentence.split())
        for word, count in counts.items():
            if count >= self.word_count_threshold:
                self.word_count[word] = count

        self.vocab_size = len(self.word_count)

        # Create word-index and index-word maps
        self.word_index = {}  # dictionary of word index
        for i, word in enumerate(self.word_count):
            self.word_index[word] = i

        self.index_word = dict(zip(self.word_index.values(), self.word_index.keys()))  # reverse mapping

        self.T = []  # target words embeddings
        self.C = []  # context words embeddings

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word

        Returns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default

        if w1 in self.word_index and w2 in self.word_index and w1 != w2:

            v1 = self.T[:, self.word_index[w1]]  # get the vector of w1
            v2 = self.T[:, self.word_index[w2]]  # get the vector of w2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:  # check if the norm is not zero
                # cosine similarity calculation
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        return sim  # default

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        closest_words = []

        if w in self.word_index:
            # Compute similarity between the specified word and all other words
            similarities = [(other_word, self.compute_similarity(w, other_word))
                            for other_word in self.word_index if other_word != w]

            # Sort the words based on their similarity scores in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Select the top n words
            closest_words = [word for word, _ in similarities[:n]]

        return closest_words

    def generate_training_data(self):
        """
        generates the training data for the skip-gram model with negative sampling.

        Returns:
            training_data: a list of tuples (target, context, label) where label is 1 for positive samples and
            0 for negative samples.

            learning_vector: a list of tuples (word, context_vector) where the context_vector is the sum of the
            context words embeddings.
        """

        training_data = []
        learning_vector = []
        vocab_indices = np.array(list(self.word_index.values()))  # list of word indices

        half_context = self.context // 2  # half context size on each side

        for sentence in self.sentences:
            words = sentence.split()
            for i, target_word in enumerate(words):
                if target_word not in self.word_index:  # ignore OOV words
                    continue
                target_index = self.word_index[target_word]
                start = max(0, i - half_context)  # start of the context window
                end = min(len(words), i + half_context + 1)  # end of the context window
                context_words = [words[j] for j in range(start, end) if j != i and words[j] in self.word_index]
                context_word_indices = [self.word_index[context_word] for context_word in context_words]

                # Generate positive samples and context vector
                context_vector = np.zeros(self.vocab_size, dtype=int)
                for context_word in context_words:
                    context_index = self.word_index[context_word]
                    training_data.append((target_index, context_index, 1))
                    context_vector[context_index] += 1

                # Generate negative samples more efficiently
                num_neg_samples_needed = self.neg_samples * len(context_word_indices)
                neg_samples = []
                while len(neg_samples) < num_neg_samples_needed:
                    sampled_indices = np.random.choice(vocab_indices, size=num_neg_samples_needed, replace=True)
                    valid_neg_samples = [idx for idx in sampled_indices if idx != target_index and
                                         idx not in context_word_indices]  # exclude target and context words
                    neg_samples.extend(valid_neg_samples[:num_neg_samples_needed - len(neg_samples)])

                for negative_context_word in neg_samples[:self.neg_samples]:
                    training_data.append((target_index, negative_context_word, 0))  # 0 for negative
                    context_vector[negative_context_word] -= 1

                # Add to learning vector
                learning_vector.append((target_word, context_vector))

        return training_data, learning_vector

    def forward_prop(self, word_index, T, C):
        """
        Performs a forward pass of the neural network for the SkipGram model.
        Args:
            word_index: Index of the target word.
            T: Embedding matrix for target words.
            C: Embedding matrix for context words.
        Returns:
            hidden: The hidden layer representation.
            output_layer: The output layer before activation.
            y: The output layer after applying sigmoid activation.
        """

        hidden = T[:, word_index][:, None]  # Get the target word embedding

        # Compute the output layer
        output_layer = np.dot(C, hidden)

        # Scale the predictions to be between 0 and 1
        y = sigmoid(output_layer)

        return hidden, y
        # return hidden, output_layer, y

    def compute_loss(self, true_label, predicted_label):
        """
        Calculates the loss between the true label and the predicted label.

        Args:
            true_label: True label (1 for positive, 0 for negative).
            predicted_label: Predicted probability.

        Returns:
            loss: Loss for this sample.
        """
        # epsilon = 1e-10  # Small value to prevent log(0)
        # predicted_label = np.clip(predicted_label, epsilon, 1 - epsilon)
        return -true_label * np.log(predicted_label) - (1 - true_label) * np.log(1 - predicted_label)

    def backward_prop(self, word_index, context_vector, hidden, y, T, C, step_size):
        """
        Performs a backward pass of the neural network for the SkipGram model.

        :return:  Updated T and C matrices after applying the gradient descent.
        """

        # Calculate the error
        error = y - context_vector[:, None]

        # Update the context matrix
        C -= step_size * np.dot(error, hidden.T)

        # Update the target matrix
        T[:, word_index] -= step_size * np.dot(C.T, error).flatten()

        return T, C

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        vocab_size = self.vocab_size  # todo: set to be the number of words in the model (how? how many, indeed?)
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # Generate training data
        training_data, learning_vector = self.generate_training_data()
        print("Training data generated")

        lowest_loss = np.inf
        no_improve_epochs = 0
        best_T, best_C = None, None

        for epoch in range(epochs):
            start_time = time.time()  # Record the start time of the epoch

            total_loss = 0

            for word_index, context_index, label in training_data:
                # Perform forward pass
                hidden, y = self.forward_prop(word_index, T, C)

                # Calculate loss for this sample
                loss = self.compute_loss(label, y[context_index])
                total_loss += loss

                # Perform backward pass
                T, C = self.backward_prop(word_index, label, hidden, y, T, C, step_size)

            # Calculate the average loss for this epoch
            avg_loss = total_loss / len(training_data)
            end_time = time.time()  # Record the end time of the epoch
            epoch_time = end_time - start_time

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f} sec")

            # Early stopping check
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                no_improve_epochs = 0
                best_T, best_C = T.copy(), C.copy()  # Save the best model
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stopping:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Use the best model found during training
        if best_T is not None and best_C is not None:
            T, C = best_T, best_C

        # Backup the last trained model (after training loop)
        self.T = T
        self.C = C

        # Save the final model
        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump((T, C), file)

        print(f"Model saved to path: '{model_path}'")

        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        if combo == 0:
            V = T
        elif combo == 1:
            V = C.T
        elif combo == 2:
            V = (T + C.T) / 2
        elif combo == 3:
            V = T + C.T
        elif combo == 4:
            V = np.concatenate((T, C.T), axis=1)  # double the dimension of the embedding space
        else:
            raise ValueError('Invalid combination mode')

        if model_path:
            with open(model_path, 'wb') as f:
                pickle.dump(V, f)

        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index:
            return None

        # Get the embeddings for each word
        v1 = self.T[:, self.word_index[w1]]
        v2 = self.T[:, self.word_index[w2]]
        v3 = self.T[:, self.word_index[w3]]

        # Calculate the vector for the missing word in the analogy
        analogy_vector = v3 + (v2 - v1)

        # Normalize the analogy vector
        analogy_vector /= np.linalg.norm(analogy_vector)

        # Calculate cosine similarities between the analogy vector and all other vectors
        similarities = np.dot(self.T.T, analogy_vector) / (
                np.linalg.norm(self.T, axis=0) * np.linalg.norm(analogy_vector))

        # Set the similarities of w1, w2, and w3 to -inf to prevent choosing them
        for word in [w1, w2, w3]:
            similarities[self.word_index[word]] = -np.inf

        # Use argmax to find the index of the word with the highest cosine similarity
        most_similar_idx = np.argmax(similarities)

        # Return the word corresponding to this index
        return self.index_word[most_similar_idx]

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        if any(word not in self.word_index for word in [w1, w2, w3, w4]):
            return False

        # Find the best match for the analogy
        best_match = self.find_analogy(w1, w2, w3)
        if best_match is None:
            return False

        # Check if the best match is the same as w4
        if best_match == w4:
            return True

        # Get the n closest words to the best match from find_analogy
        closest_words = self.get_closest_words(best_match, n=n)

        # Check if w4 is among these words
        return w4 in closest_words
