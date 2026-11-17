"""Lecture 4 hierarchical Tsetlin Machine for two posts.

PyHierarchicalTsetlinMachineCUDA needs an NVIDIA GPU. This computer has none,
so the same tree is trained here. It is not the CUDA library.

One clause is a tree with two groups, one for Post A and one for Post B.
Inside a group, each leaf is an ordinary conjunction, and the group adds the
leaf outputs. That is OR replaced by addition. The clause multiplies the two
group sums. That is AND. A leaf is updated only when the other group is
active, which is the AND on the path from that leaf to the root.

s stays 1.0. With the boost flag off, Type I never includes a true literal.
With the boost flag on, a firing leaf includes every true literal. Both
settings are trained so the report can compare them.
"""

from __future__ import annotations

import numpy as np


class HierarchicalTsetlinMachine:
    """Two classes. Each clause is an AND of a Post A group and a Post B group."""

    def __init__(
        self,
        n_words: int,
        n_clauses: int,
        n_leaves: int,
        T: int,
        s: float = 1.0,
        number_of_state_bits: int = 8,
        seed: int = 457,
        boost_true_positive_feedback: int = 0,
    ) -> None:
        if n_words < 1:
            raise ValueError("Need at least one word.")
        if n_clauses < 2 or n_clauses % 2 != 0:
            raise ValueError("The clause count must be a positive even number.")
        if n_leaves < 1:
            raise ValueError("Each group needs at least one leaf.")
        if T < 1:
            raise ValueError("T must be at least 1.")
        if s < 1:
            raise ValueError("s must be at least 1.")
        if boost_true_positive_feedback not in (0, 1):
            raise ValueError("boost_true_positive_feedback must be 0 or 1.")
        self.n_words = int(n_words)
        self.n_features = self.n_words * 2
        self.n_clauses = int(n_clauses)
        self.n_leaves = int(n_leaves)
        self.n_groups = 2
        self.T = int(T)
        self.s = float(s)
        self.boost_true_positive_feedback = int(boost_true_positive_feedback)
        self.n_classes = 2
        self.include_threshold = 1 << (number_of_state_bits - 1)
        self.max_state = (1 << number_of_state_bits) - 1
        self.rng = np.random.default_rng(seed)
        # class, clause, group, leaf, literal. Literals are the post's own [x, not x].
        self.state = np.full(
            (self.n_classes, self.n_clauses, self.n_groups, self.n_leaves, self.n_words * 2),
            self.include_threshold - 1,
            dtype=np.uint8,
        )
        self.polarity = np.where(np.arange(self.n_clauses) % 2 == 0, 1, -1).astype(np.int16)

    def _post_literals(self, post: np.ndarray) -> np.ndarray:
        row = np.asarray(post, dtype=np.uint8)
        return np.concatenate((row, 1 - row)).astype(bool)

    def _leaf_output(self, state: np.ndarray, literal_true: np.ndarray, predict: bool) -> np.ndarray:
        """state is (clauses, leaves, literals). Returns (clauses, leaves)."""

        included = state >= self.include_threshold
        false_index = np.flatnonzero(~literal_true)
        if len(false_index) == 0:
            output = np.ones(state.shape[:2], dtype=bool)
        else:
            output = ~(included[:, :, false_index]).any(axis=2)
        if predict:
            output = output & included.any(axis=2)
        return output

    def _clause_parts(self, features: np.ndarray, predict: bool) -> list[list[np.ndarray]]:
        """For each class, the two group sums. Each sum has shape (rows, clauses)."""

        rows = np.asarray(features, dtype=np.uint8)
        if rows.ndim != 2 or rows.shape[1] != self.n_features:
            raise ValueError(f"Expected rows with {self.n_features} features. Got {rows.shape}.")
        parts: list[list[np.ndarray]] = []
        posts = (rows[:, : self.n_words], rows[:, self.n_words :])
        for class_id in range(self.n_classes):
            groups = []
            for group, post in enumerate(posts):
                sums = np.zeros((len(rows), self.n_clauses), dtype=np.int32)
                for start in range(0, len(rows), 256):
                    block = post[start : start + 256]
                    literal_true = np.concatenate((block, 1 - block), axis=1).astype(bool)
                    included = self.state[class_id, :, group] >= self.include_threshold
                    mask = included.reshape(-1, self.n_words * 2)
                    literal_false = (1 - literal_true.astype(np.float32))
                    violations = literal_false @ mask.T.astype(np.float32)
                    fired = (violations == 0)
                    if predict:
                        fired = fired & mask.any(axis=1)[None, :]
                    fired = fired.reshape(len(block), self.n_clauses, self.n_leaves)
                    sums[start : start + len(block)] = fired.sum(axis=2)
                groups.append(sums)
            parts.append(groups)
        return parts

    def class_scores(self, features: np.ndarray) -> np.ndarray:
        parts = self._clause_parts(features, predict=True)
        scores = np.zeros((len(features), self.n_classes), dtype=np.int32)
        for class_id, groups in enumerate(parts):
            product = groups[0].astype(np.int32) * groups[1].astype(np.int32)
            total = (product * self.polarity[None, :]).sum(axis=1)
            scores[:, class_id] = np.clip(total, -self.T, self.T)
        return scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.class_scores(features), axis=1).astype(np.uint32)

    def fit_epoch(self, features: np.ndarray, targets: np.ndarray) -> None:
        rows = np.asarray(features, dtype=np.uint8)
        labels = np.asarray(targets, dtype=np.uint32)
        exclude_probability = 1.0 / self.s
        if self.boost_true_positive_feedback:
            include_probability = 1.0
        else:
            include_probability = (self.s - 1.0) / self.s
        for row, target in zip(rows, labels):
            posts = (row[: self.n_words], row[self.n_words :])
            literals = [self._post_literals(post) for post in posts]
            for class_id in range(self.n_classes):
                class_target = 1 if class_id == int(target) else 0
                self._update_class(class_id, literals, class_target, include_probability, exclude_probability)

    def _update_class(
        self,
        class_id: int,
        literals: list[np.ndarray],
        target: int,
        include_probability: float,
        exclude_probability: float,
    ) -> None:
        state = self.state[class_id]
        outputs = [
            self._leaf_output(state[:, group], literals[group], predict=False) for group in range(self.n_groups)
        ]
        group_sum = [output.sum(axis=1) for output in outputs]
        product = group_sum[0].astype(np.int32) * group_sum[1].astype(np.int32)
        vote = int(np.clip(int((self.polarity * product.astype(np.int16)).sum()), -self.T, self.T))
        probability = (self.T + (1 - 2 * target) * vote) / (2.0 * self.T)
        selected = self.rng.random(self.n_clauses) <= probability
        if not selected.any():
            return
        positive = self.polarity > 0
        type_i = selected & (positive if target == 1 else ~positive)
        type_ii = selected & ~type_i & (product > 0)
        for group in range(self.n_groups):
            other = 1 - group
            # The other group is the AND on the path to the root.
            active = group_sum[other] > 0
            self._update_leaves(
                state[:, group],
                outputs[group],
                literals[group],
                type_i & active,
                type_ii & active,
                include_probability,
                exclude_probability,
            )

    def _update_leaves(
        self,
        state: np.ndarray,
        output: np.ndarray,
        literal_true: np.ndarray,
        type_i: np.ndarray,
        type_ii: np.ndarray,
        include_probability: float,
        exclude_probability: float,
    ) -> None:
        """state and output are (clauses, leaves, ...) and (clauses, leaves)."""

        true_index = np.flatnonzero(literal_true)
        false_index = np.flatnonzero(~literal_true)
        type_ii_clauses = np.flatnonzero(type_ii)
        if len(type_ii_clauses) and len(false_index):
            leaves = output[type_ii_clauses]
            for leaf in range(self.n_leaves):
                chosen = type_ii_clauses[leaves[:, leaf]]
                if len(chosen) == 0:
                    continue
                cells = np.ix_(chosen, [leaf], false_index)
                block = state[cells]
                state[cells] = block + (block < self.include_threshold)

        if not type_i.any():
            return
        for leaf in range(self.n_leaves):
            firing = np.flatnonzero(type_i & output[:, leaf])
            silent = np.flatnonzero(type_i & ~output[:, leaf])
            if len(firing) and len(false_index):
                cells = np.ix_(firing, [leaf], false_index)
                block = state[cells].astype(np.int16)
                state[cells] = np.maximum(block - self._hits(block.shape, exclude_probability), 0).astype(np.uint8)
                if include_probability > 0 and len(true_index):
                    cells = np.ix_(firing, [leaf], true_index)
                    block = state[cells].astype(np.int16)
                    state[cells] = np.minimum(
                        block + self._hits(block.shape, include_probability), self.max_state
                    ).astype(np.uint8)
            if len(silent):
                block = state[silent, leaf].astype(np.int16)
                state[silent, leaf] = np.maximum(
                    block - self._hits(block.shape, exclude_probability), 0
                ).astype(np.uint8)

    def _hits(self, shape: tuple[int, ...], probability: float) -> np.ndarray | int:
        if probability >= 1:
            return 1
        return (self.rng.random(shape) <= probability).astype(np.int16)

    def describe(self, vocabulary: list[str], limit_clauses: int = 2, limit_literals: int = 6) -> str:
        lines = [
            "Each clause is the AND of a Post A group and a Post B group.",
            "A group adds its leaf conjunctions. s = 1.0. "
            f"Boost is {'on' if self.boost_true_positive_feedback else 'off'}.",
            "",
        ]
        for class_id, title in ((1, "same category"), (0, "different category")):
            lines.append(f"Class {class_id}, {title}:")
            shown = 0
            for clause_id in range(self.n_clauses):
                if shown >= limit_clauses:
                    break
                groups = []
                nonempty = False
                for group, post_name in ((0, "Post A"), (1, "Post B")):
                    leaves = []
                    for leaf in range(self.n_leaves):
                        included = np.flatnonzero(
                            self.state[class_id, clause_id, group, leaf] >= self.include_threshold
                        )
                        if len(included) == 0:
                            continue
                        nonempty = True
                        phrases = [
                            _leaf_phrase(int(index), self.n_words, vocabulary, post_name)
                            for index in included[:limit_literals]
                        ]
                        more = f", and {len(included) - limit_literals} more" if len(included) > limit_literals else ""
                        leaves.append(", ".join(phrases) + more)
                    groups.append(f"  {post_name}: " + (" | ".join(leaves) if leaves else "(empty)"))
                if not nonempty:
                    continue
                shown += 1
                kind = "positive" if clause_id % 2 == 0 else "negative"
                lines.append(f" clause {clause_id} ({kind})")
                lines.extend(groups)
            if shown == 0:
                lines.append(" no included literals yet")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _leaf_phrase(literal_index: int, n_words: int, vocabulary: list[str], post_name: str) -> str:
    negated = literal_index >= n_words
    feature = literal_index - n_words if negated else literal_index
    word = vocabulary[feature]
    if negated:
        return f"{word} absent from {post_name}"
    return f"{word} present in {post_name}"


def learns_a_pair_pattern() -> float:
    """A hand-built tree votes only when both posts contain the same word."""

    machine = HierarchicalTsetlinMachine(n_words=2, n_clauses=2, n_leaves=1, T=5, s=1.0, seed=1)
    # Include "word 0 present" on one positive leaf in each post, class 1.
    present = 0
    machine.state[1, 0, 0, 0, present] = machine.include_threshold
    machine.state[1, 0, 1, 0, present] = machine.include_threshold
    rows = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    scores = machine.class_scores(rows)
    # Both posts contain word 0 in the first two rows, so class 1 gets the product 1.
    return float(np.array_equal(scores[:, 1], np.array([1, 1, 0, 0], dtype=np.int32)))
