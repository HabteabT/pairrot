"""Vanilla multiclass Tsetlin Machine used by the Pairrot pilot.

The PyPI package ``tmu`` 0.7.1 did not install here: building its C extension
stopped with "Microsoft Visual C++ 14.0 or greater is required". This file is
the standard vanilla machine, written so the feedback can be read directly.

The update rules follow ``tm_update`` in pyTsetlinMachine's
``ConvolutionalTsetlinMachine.c`` for one patch, unweighted clauses, and
``boost_true_positive_feedback = 0``:

- Even clauses vote for the class. Odd clauses vote against it.
- A clause outputs 1 when every included literal is true.
- During prediction an empty clause outputs 0. During an update it outputs 1,
  which is the library's ``PREDICT`` versus ``UPDATE`` distinction.
- The class vote is clipped to ``[-T, T]``. A clause is updated with
  probability ``(T - vote) / (2T)`` for the target class and
  ``(T + vote) / (2T)`` for every other class.
- Type I is given to clauses whose polarity agrees with that target bit.
  Type II is given to the opposite polarity, and only when the clause outputs 1.
- With the boost flag off, a true literal is reinforced toward Include with
  probability ``(s - 1) / s``. A false literal is reinforced toward Exclude
  with probability ``1 / s``. At ``s = 1`` those probabilities are 0 and 1.
- Type II reinforces Include for literals that are false in the current
  example and are not already included. That is the elimination step: the
  clause gains a literal that blocks this example.

``number_of_state_bits = 8`` matches the library default. Include is the
high bit, so the state runs from 0 to 255 and Include begins at 128. New
automata start at 127, one step on the Exclude side of that boundary.
"""

from __future__ import annotations

import numpy as np


class VanillaTsetlinMachine:
    """One vanilla Tsetlin Machine for labels 0, 1, ..., n_classes - 1."""

    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        T: int,
        s: float,
        n_classes: int = 2,
        number_of_state_bits: int = 8,
        seed: int = 457,
    ) -> None:
        if n_clauses < 2 or n_clauses % 2 != 0:
            raise ValueError("The clause count must be a positive even number.")
        if T < 1:
            raise ValueError("T must be at least 1.")
        if s < 1:
            raise ValueError("s must be at least 1.")
        if number_of_state_bits < 1:
            raise ValueError("number_of_state_bits must be at least 1.")

        self.n_features = int(n_features)
        self.n_clauses = int(n_clauses)
        self.T = int(T)
        self.s = float(s)
        self.n_classes = int(n_classes)
        self.number_of_state_bits = int(number_of_state_bits)
        self.include_threshold = 1 << (self.number_of_state_bits - 1)
        self.max_state = (1 << self.number_of_state_bits) - 1
        self.boost_true_positive_feedback = 0
        self.weighted_clauses = False
        self.rng = np.random.default_rng(seed)
        # Shape: class, clause, literal. Literals are [x, not x].
        self.state = np.full(
            (self.n_classes, self.n_clauses, self.n_features * 2),
            self.include_threshold - 1,
            dtype=np.uint8,
        )
        self.polarity = np.where(np.arange(self.n_clauses) % 2 == 0, 1, -1).astype(np.int16)

    def literals(self, features: np.ndarray) -> np.ndarray:
        """Truth of each literal. The second half is the negated feature."""

        row = np.asarray(features, dtype=np.uint8)
        return np.concatenate((row, 1 - row)).astype(bool)

    def clause_output(self, class_id: int, literal_true: np.ndarray, predict: bool) -> np.ndarray:
        included = self.state[class_id] >= self.include_threshold
        satisfied = (~included) | literal_true
        output = satisfied.all(axis=1)
        if predict:
            output = output & included.any(axis=1)
        return output

    def vote(self, class_id: int, literal_true: np.ndarray, predict: bool) -> int:
        output = self.clause_output(class_id, literal_true, predict=predict)
        total = int((self.polarity * output.astype(np.int16)).sum())
        return int(np.clip(total, -self.T, self.T))

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = np.empty(len(features), dtype=np.uint32)
        for index, row in enumerate(features):
            literal_true = self.literals(row)
            scores = [self.vote(class_id, literal_true, predict=True) for class_id in range(self.n_classes)]
            # The lowest class index wins a tie. A 0-0 vote therefore predicts class 0.
            predictions[index] = int(np.argmax(scores))
        return predictions

    def predict_scores(self, row: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return clipped class votes and the predicting clause outputs."""

        literal_true = self.literals(row)
        outputs = [
            self.clause_output(class_id, literal_true, predict=True) for class_id in range(self.n_classes)
        ]
        scores = np.array(
            [
                int(np.clip(int((self.polarity * output.astype(np.int16)).sum()), -self.T, self.T))
                for output in outputs
            ],
            dtype=np.int16,
        )
        return scores, outputs

    def fit(self, features: np.ndarray, targets: np.ndarray, epochs: int) -> None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        for _ in range(epochs):
            for row, target in zip(features, targets):
                literal_true = self.literals(row)
                for class_id in range(self.n_classes):
                    class_target = 1 if class_id == int(target) else 0
                    self._update_class(class_id, literal_true, class_target)

    def _update_class(self, class_id: int, literal_true: np.ndarray, target: int) -> None:
        vote = self.vote(class_id, literal_true, predict=False)
        probability = (self.T + (1 - 2 * target) * vote) / (2.0 * self.T)
        selected = self.rng.random(self.n_clauses) <= probability
        if not selected.any():
            return

        output = self.clause_output(class_id, literal_true, predict=False)
        positive = self.polarity > 0
        type_i = selected & (positive if target == 1 else ~positive)
        type_ii = selected & ~type_i & output
        state = self.state[class_id]

        if type_ii.any():
            grow = type_ii[:, None] & (~literal_true) & (state < self.include_threshold)
            state[grow] = np.minimum(state[grow].astype(np.uint16) + 1, self.max_state).astype(np.uint8)

        if type_i.any():
            include_probability = (self.s - 1.0) / self.s
            exclude_probability = 1.0 / self.s
            if include_probability > 0:
                include_hit = self.rng.random((self.n_clauses, self.n_features * 2)) <= include_probability
            else:
                include_hit = np.zeros((self.n_clauses, self.n_features * 2), dtype=bool)
            if exclude_probability >= 1:
                exclude_hit = np.ones((self.n_clauses, self.n_features * 2), dtype=bool)
            else:
                exclude_hit = self.rng.random((self.n_clauses, self.n_features * 2)) <= exclude_probability

            clause_on = type_i & output
            clause_off = type_i & ~output
            grow = (
                clause_on[:, None]
                & literal_true
                & include_hit
            )
            shrink = (clause_on[:, None] & (~literal_true) & exclude_hit) | (
                clause_off[:, None] & exclude_hit
            )
            state[grow] = np.minimum(state[grow].astype(np.uint16) + 1, self.max_state).astype(np.uint8)
            state[shrink] = np.maximum(state[shrink].astype(np.int16) - 1, 0).astype(np.uint8)

    def included_indices(self, class_id: int, clause_id: int) -> np.ndarray:
        return np.flatnonzero(self.state[class_id, clause_id] >= self.include_threshold)


def literal_phrase(literal_index: int, n_features: int, vocabulary: list[str]) -> str:
    """Map one automaton back to a Post A or Post B word.

    The feature layout is Post A, then Post B. Each feature has a positive
    literal and, after ``n_features``, its negation.
    """

    negated = literal_index >= n_features
    feature = literal_index - n_features if negated else literal_index
    vocabulary_size = len(vocabulary)
    if feature < vocabulary_size:
        post = "Post A"
        word = vocabulary[feature]
    else:
        post = "Post B"
        word = vocabulary[feature - vocabulary_size]
    if negated:
        return f"{word} absent from {post}"
    return f"{word} present in {post}"


def verify_s_equals_one() -> list[str]:
    """Check the s=1.0 feedback boundary on one hand-written example.

    The example is illustrative. It is not a 20 Newsgroups pair.
    """

    machine = VanillaTsetlinMachine(n_features=2, n_clauses=2, T=1, s=1.0, seed=1)
    # One true feature and one false feature. Literals: x0, x1, not x0, not x1.
    literal_true = np.array([True, False, False, True])
    before = machine.state[0, 0].copy()
    machine._update_class(0, literal_true, target=1)
    # Clause 0 is positive, so target 1 gives it Type I. With vote starting at 0
    # and T = 1, its update probability is 1/2, so repeat until it has moved.
    attempts = 0
    while np.array_equal(machine.state[0, 0], before):
        machine._update_class(0, literal_true, target=1)
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Type I feedback did not touch the positive clause.")

    after = machine.state[0, 0]
    if int(after[0]) != int(before[0]) or int(after[3]) != int(before[3]):
        raise RuntimeError(
            "At s=1.0, Type I changed a true literal. "
            f"Before {before.tolist()}, after {after.tolist()}."
        )
    if int(after[1]) >= int(before[1]) or int(after[2]) >= int(before[2]):
        raise RuntimeError(
            "At s=1.0, Type I did not reinforce Exclude on the false literals. "
            f"Before {before.tolist()}, after {after.tolist()}."
        )
    if np.any(after >= machine.include_threshold):
        raise RuntimeError("Type I at s=1.0 included a literal.")

    eliminator = VanillaTsetlinMachine(n_features=2, n_clauses=2, T=1, s=1.0, seed=2)
    before_negative = eliminator.state[0, 1].copy()
    moved = False
    for _ in range(100):
        eliminator._update_class(0, literal_true, target=1)
        if not np.array_equal(eliminator.state[0, 1], before_negative):
            moved = True
            break
    if not moved:
        raise RuntimeError("Type II feedback did not touch the negative clause.")
    included = eliminator.included_indices(0, 1)
    # False literals are indexes 1 (x1) and 2 (not x0). Those are the ones Type II may include.
    if not set(included.tolist()).issubset({1, 2}) or len(included) == 0:
        raise RuntimeError(
            "Type II did not include a currently false literal. "
            f"Included {included.tolist()}."
        )

    empty = VanillaTsetlinMachine(n_features=2, n_clauses=2, T=1, s=1.0, seed=3)
    training_output = empty.clause_output(0, literal_true, predict=False)
    predicting_output = empty.clause_output(0, literal_true, predict=True)
    if not training_output.all() or predicting_output.any():
        raise RuntimeError("An empty clause did not follow the update/predict output rule.")

    return [
        "s=1.0 Type I leaves true literals unchanged and reinforces Exclude on false literals",
        "s=1.0 Type I does not include a literal",
        "Type II includes a literal that is false in the current example",
        "an empty clause outputs 1 during an update and 0 during prediction",
    ]
