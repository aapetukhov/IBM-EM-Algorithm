from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


def get_indices(sentence_pair: TokenizedSentencePair) -> np.ndarray:
    """Gets the required indices from sentence pair indices

    Args:
        sentence_pair (TokenizedSentencePair): pair of source and target sentences

    Returns:
        np.ndarray: indices to take from self.translation_probs
    """
    return np.ix_(sentence_pair.source_tokens, sentence_pair.target_tokens)


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(
        self, sentences: List[TokenizedSentencePair]
    ) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (
            2
            * self.cooc.astype(np.float32)
            / (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True))
        )

    def align(self, sentences):
        elbo = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                enumerate(sentence.source_tokens, 1),
                enumerate(sentence.target_tokens, 1),
            ):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            elbo.append(alignment)
        return elbo


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full(
            (num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32
        )  # theta matrix
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[k][j][i] gives a posterior
            probability of target token i to be aligned to source token j in a sentence k.
            в моих обозначениях это delta(k, i, j)
        """
        posteriors = []
        for sentence_pair in parallel_corpus:
            translation_probs = self.translation_probs[get_indices(sentence_pair)]
            # нормируем вероятности к единичной сумме
            posteriors.append(
                translation_probs / np.sum(translation_probs, axis=0, keepdims=True)
            )

        return posteriors

    def _compute_elbo(
        self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]
    ) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0

        for posterior, sentence_pair in zip(posteriors, parallel_corpus):
            elbo += np.sum(
                posterior
                * (
                    (
                        np.log(
                            np.maximum(
                                self.translation_probs[get_indices(sentence_pair)],
                                1e-10,
                            )
                        )
                    )
                    - np.log(
                        np.maximum(posterior * len(sentence_pair.source_tokens), 1e-10)
                    )
                )
            )

        return elbo

    def _m_step(
        self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]
    ):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs.fill(0)

        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, get_indices(sentence_pair), posterior)

        self.translation_probs /= self.translation_probs.sum(axis=1, keepdims=True)

        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for _ in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        posteriors = self._e_step(sentences)
        alignments = []

        for posterior in posteriors:
            alignments.append(
                [
                    (source_pos, target_pos)
                    for target_pos, source_pos in enumerate(
                        posterior.argmax(axis=0) + 1, 1
                    )
                ]
            )
        return alignments


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        if (src_length, tgt_length) not in self.alignment_probs.keys():
            self.alignment_probs[(src_length, tgt_length)] = np.full(
                shape=(src_length, tgt_length), fill_value=1 / src_length
            )

        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence_pair in parallel_corpus:
            translation_probs = self.translation_probs[get_indices(sentence_pair)]
            alignment_probs = self._get_probs_for_lengths(
                len(sentence_pair.source_tokens), len(sentence_pair.target_tokens)
            )
            posterior = translation_probs * alignment_probs
            # нормируем вероятности к единичной сумме
            posteriors.append(posterior / np.sum(posterior, axis=0, keepdims=True))

        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbo = 0
        for posterior, sentence_pair in zip(posteriors, parallel_corpus):
            elbo += np.sum(
                posterior
                * (
                    (
                        np.log(
                            np.maximum(
                                self.translation_probs[get_indices(sentence_pair)]
                                * self._get_probs_for_lengths(
                                    len(sentence_pair.source_tokens),
                                    len(sentence_pair.target_tokens),
                                ),
                                1e-10,
                            )
                        )
                    )
                    - np.log(np.maximum(posterior, 1e-10))
                )
            )

        return elbo

    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs.fill(0)
        self.alignment_probs = dict()

        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, get_indices(sentence_pair), posterior)
            if posterior.shape in self.alignment_probs.keys():
                self.alignment_probs[posterior.shape] += posterior
            else:
                self.alignment_probs[posterior.shape] = posterior

        self.translation_probs /= np.sum(self.translation_probs, axis=1, keepdims=True)

        for alignment_prob in self.alignment_probs:
            alignment_prob /= np.sum(alignment_prob, axis=0, keepdims=True)

        return self._compute_elbo(parallel_corpus, posteriors)


class WordPositionAlignerWithPriors(WordPositionAligner):
    def __init__(
        self,
        num_source_words,
        num_target_words,
        num_iters,
        initial_lambda: float = 4.0,
        learning_rate: float = 0.01,
        grad_reg: float = 1e-1,
    ):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.lambda_ = initial_lambda  # controls the decay of alignment probabilities when moving away from the diagonal
        self.learning_rate = learning_rate
        self.grad_reg = grad_reg

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        if (src_length, tgt_length) not in self.alignment_probs:
            log_probs = np.zeros((src_length, tgt_length))
            for i in range(tgt_length):
                log_Z = self.log_Z_lambda(i, tgt_length, src_length)
                for j in range(src_length):
                    log_probs[j, i] = (
                        self.lambda_ * self.h(i, j, tgt_length, src_length) - log_Z
                    )
            self.alignment_probs[(src_length, tgt_length)] = np.exp(log_probs)
        return self.alignment_probs[(src_length, tgt_length)]

    def h(self, i: int, j: int, m: int, n: int):
        """
        Calculates the deviation from the diagonal.

        Parameters:
        i: index in the target sentence
        j: index in the source sentence
        m: length of the target sentence
        n: length of the source sentence
        """

        return -abs(i / m - j / n)

    def log_Z_lambda(self, i, m, n):
        """Calculates the log of the normalizing coefficient Z_lambda(i, m, n) using a logarithmic approach."""
        j_up = int(np.floor(i * n / m))
        j_down = j_up + 1
        log_r = -self.lambda_ / n

        log_term1 = log_r * (j_up + 1) + self.lambda_ * self.h(i, j_up, m, n)
        log_term2 = log_r * (n - j_down) + self.lambda_ * self.h(i, j_down, m, n)

        return np.logaddexp(log_term1, log_term2)

    # NOT RECOMMENDED TO USE THIS FUNCTION (only for single-time call)
    def delta(self, i, j, m, n):
        """calculates the probability of alignment delta(i, j, m, n)"""
        if 0 <= j <= n:
            return np.exp(self.lambda_ * self.h(i, j, m, n)) / self.Z_lambda(i, m, n)
        else:
            return 0

    def update_translation_probs(self, parallel_corpus, posteriors):
        """Updating translation probabilities on M-step."""
        count_ef = np.zeros_like(self.translation_probs)
        total_f = np.zeros((self.num_source_words,))

        for idx, sentence_pair in enumerate(parallel_corpus):
            posterior = posteriors[idx]
            for i, source_token in enumerate(sentence_pair.source_tokens):
                for j, target_token in enumerate(sentence_pair.target_tokens):
                    count_ef[source_token, target_token] += posterior[i, j]
                    total_f[source_token] += posterior[i, j]

        self.translation_probs = count_ef / total_f[:, np.newaxis]

    # NOT RECOMMENDED TO USE THIS FUNCTION AT ALl
    def update_lambda(self, parallel_corpus, posteriors):
        """Updating lambda on M-step using gradient descent."""
        gradient = 0.0

        for idx, sentence_pair in enumerate(parallel_corpus):
            posterior = posteriors[idx]
            m, n = len(sentence_pair.source_tokens), len(sentence_pair.target_tokens)

            for i in range(m):
                # Expected value of h under E-step posterior
                expected_h = sum(posterior[i, j] * self.h(i, j, m, n) for j in range(n))

                # Calculating j_up and j_down for the partition function and its derivative
                j_up = int(np.floor(i * n / m))
                j_down = j_up + 1

                # Calculating Z_lambda for normalization
                Z_lambda = self.Z_lambda(i, m, n)

                # Compute t_j_up and t_j_down based on the formula provided
                t_j_up = np.exp(self.lambda_ * self.h(i, j_up, m, n))
                t_j_down = np.exp(self.lambda_ * self.h(i, j_down, m, n))

                # Adding terms to the gradient
                gradient += (
                    expected_h
                    - (
                        t_j_up * self.h(i, j_up, m, n)
                        + t_j_down * self.h(i, j_down, m, n)
                    )
                    / Z_lambda
                )

        # Update lambda using gradient ascent
        self.lambda_ += self.learning_rate * gradient

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence_pair in parallel_corpus:
            log_translation_probs = np.log(
                self.translation_probs[get_indices(sentence_pair)]
            )
            log_alignment_probs = np.log(
                self._get_probs_for_lengths(
                    len(sentence_pair.source_tokens), len(sentence_pair.target_tokens)
                )
            )
            log_posterior = log_translation_probs + log_alignment_probs
            log_posterior -= np.max(
                log_posterior, axis=0, keepdims=True
            )  # For numerical stability
            posterior = np.exp(log_posterior)
            posterior /= np.sum(posterior, axis=0, keepdims=True)
            posteriors.append(posterior)
        return posteriors

    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs.fill(0)
        count_ef = np.zeros_like(self.translation_probs)
        total_f = np.zeros((self.num_source_words,))
        lambda_gradient = 0.0

        for sentence_pair, posterior in zip(parallel_corpus, posteriors):
            n, m = len(sentence_pair.source_tokens), len(sentence_pair.target_tokens)

            # Update translation probs
            for j, source_token in enumerate(sentence_pair.source_tokens):
                for i, target_token in enumerate(sentence_pair.target_tokens):
                    count_ef[source_token, target_token] += posterior[j, i]
                    total_f[source_token] += posterior[j, i]

            # Update lambda
            for i in range(m):
                expected_h = sum(posterior[j, i] * self.h(i, j, m, n) for j in range(n))
                j_up = int(np.floor(i * n / m))
                j_down = j_up + 1
                Z_lambda = np.exp(self.log_Z_lambda(i, m, n))
                t_j_up = np.exp(self.lambda_ * self.h(i, j_up, m, n))
                t_j_down = np.exp(self.lambda_ * self.h(i, j_down, m, n))
                lambda_gradient += (
                    expected_h
                    - (
                        t_j_up * self.h(i, j_up, m, n)
                        + t_j_down * self.h(i, j_down, m, n)
                    )
                    / Z_lambda
                )

        # Update translation probabilities
        self.translation_probs = count_ef / total_f[:, np.newaxis]

        if np.isnan(lambda_gradient) or np.isinf(lambda_gradient):
            print(f"Invalid gradient detected: {lambda_gradient}")

        lambda_gradient *= self.grad_reg

        # Update lambda
        self.lambda_ = np.clip(
            self.lambda_ + self.learning_rate * lambda_gradient, -500, 100
        )  # Clip lambda to avoid extreme values

        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for iteration in range(self.num_iters):
            print(f"Iteration {iteration}")
            print(f"Lambda: {self.lambda_}")
            print("-" * 10 + "\n")
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history
