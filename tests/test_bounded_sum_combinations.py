from reasoning_gym.arithmetic.bounded_sum_combinations import (
    BoundedSumCombinationsConfig,
    BoundedSumCombinationsDataset,
)

def test_bounded_sum_combinations_single():
    config = BoundedSumCombinationsConfig(mode="single", seed=42, size=10)
    dataset = BoundedSumCombinationsDataset(config)
    item = dataset[0]

    assert "question" in item
    assert "answer" in item
    assert dataset.score_answer(item["answer"], item) == 1.0

def test_bounded_sum_combinations_all():
    config = BoundedSumCombinationsConfig(mode="all", seed=42, size=10)
    dataset = BoundedSumCombinationsDataset(config)
    item = dataset[0]

    assert dataset.score_answer(item["answer"], item) == 1.0