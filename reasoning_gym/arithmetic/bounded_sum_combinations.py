import random
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "bounded_sum_combinations"


@dataclass
class BoundedSumCombinationsConfig:
    """Configuration for bounded sum combinations task generation"""

    min_numbers: int = 3
    max_numbers: int = 6
    min_value: int = 1
    max_value: int = 20
    min_usage: int = 1
    max_usage: int = 3
    mode: str = "single"  # Options: "single" (any 1 valid combo) or "all" (all unique combos)
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert self.min_numbers > 0, "min_numbers must be positive"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert self.min_usage > 0, "min_usage must be positive"
        assert self.max_usage >= self.min_usage, "max_usage must be >= min_usage"
        assert self.mode in ("single", "all"), "mode must be either 'single' or 'all'"


class BoundedSumCombinationsDataset(ProceduralDataset):
    """Generates bounded sum combination problems"""

    def __init__(self, config: BoundedSumCombinationsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single task

        Returns dict with:
            - question: str formatted task prompt
            - answer: str ground truth (single expression or line-separated list of expressions)
            - metadata: dict with structural attributes and internal solution list
        """
        rng = random.Random(self.seed + idx)

        # Generate unique available numbers and their max usage limits
        num_candidates = rng.randint(self.config.min_numbers, self.config.max_numbers)
        numbers = sorted(rng.sample(range(self.config.min_value, self.config.max_value + 1), num_candidates))
        max_uses = [rng.randint(self.config.min_usage, self.config.max_usage) for _ in numbers]

        # Find all valid combinations via backtracking
        all_combinations = self._find_all_combinations(numbers, max_uses)

        # Ensure at least one solution exists by constructing a target from a randomly chosen solution
        # or choosing a target sum from existing combinations
        if not all_combinations:
            # Fallback guarantee: force at least 1 combination
            chosen_counts = [rng.randint(0, limit) for limit in max_uses]
            if sum(chosen_counts) == 0:
                chosen_counts[0] = 1
            target = sum(num * count for num, count in zip(numbers, chosen_counts))
            all_combinations = self._find_all_combinations_for_target(numbers, max_uses, target)
        else:
            # Pick a target sum from reachable sums
            all_targets = list({sum(c) for c in all_combinations})
            target = rng.choice(all_targets)
            all_combinations = self._find_all_combinations_for_target(numbers, max_uses, target)

        # Sort combinations deterministically: smaller terms first, then lexicographically
        formatted_combos = [self._format_combination(c) for c in sorted(all_combinations)]

        if self.config.mode == "single":
            question = (
                f"Available numbers: {numbers}\n"
                f"Maximum uses: {max_uses}\n"
                f"Target: {target}\n\n"
                f"Find ONE valid combination of numbers that sums to the target, respecting the maximum uses limit. "
                f"Format your answer as addition (e.g., '2 + 3 + 5 = {target}')."
            )
            answer = formatted_combos[0]
        else:
            question = (
                f"Available numbers: {numbers}\n"
                f"Maximum uses: {max_uses}\n"
                f"Target: {target}\n\n"
                f"Find ALL unique combinations of numbers that sum to the target, respecting the maximum uses limit. "
                f"Provide each combination on a new line formatted as addition (e.g., '2 + 3 + 5 = {target}')."
            )
            answer = "\n".join(formatted_combos)

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "numbers": numbers,
                "max_uses": max_uses,
                "target": target,
                "mode": self.config.mode,
                "valid_combinations": formatted_combos,
                "total_valid_solutions": len(formatted_combos),
            },
        }

    def _find_all_combinations(self, numbers: list[int], max_uses: list[int]) -> list[list[int]]:
        """Backtracking helper to explore valid combinations across all possible targets"""
        results = []

        def backtrack(index: int, current_combo: list[int]):
            if index == len(numbers):
                if current_combo:
                    results.append(list(current_combo))
                return

            num = numbers[index]
            limit = max_uses[index]
            for count in range(limit + 1):
                current_combo.extend([num] * count)
                backtrack(index + 1, current_combo)
                for _ in range(count):
                    current_combo.pop()

        backtrack(0, [])
        return results

    def _find_all_combinations_for_target(
        self, numbers: list[int], max_uses: list[int], target: int
    ) -> list[list[int]]:
        """Backtracking helper to find all unique combinations that sum to target"""
        results = []

        def backtrack(index: int, current_sum: int, current_combo: list[int]):
            if current_sum == target:
                results.append(list(current_combo))
                return
            if current_sum > target or index == len(numbers):
                return

            num = numbers[index]
            limit = max_uses[index]
            for count in range(limit + 1):
                added_sum = num * count
                if current_sum + added_sum > target:
                    break
                current_combo.extend([num] * count)
                backtrack(index + 1, current_sum + added_sum, current_combo)
                for _ in range(count):
                    current_combo.pop()

        backtrack(0, 0, [])
        return results

    @staticmethod
    def _format_combination(combo: list[int]) -> str:
        """Formats a sorted list of numbers into addition expression string"""
        terms = " + ".join(map(str, combo))
        return f"{terms} = {sum(combo)}"

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the user's/model's answer.

        For mode 'single': returns 1.0 if the answer represents ANY valid combination.
        For mode 'all': returns 1.0 if all unique valid combinations are provided.
        """
        if not answer:
            return 0.0

        valid_solutions = set(entry["metadata"]["valid_combinations"])
        numbers = entry["metadata"]["numbers"]
        max_uses_dict = dict(zip(numbers, entry["metadata"]["max_uses"]))
        target = entry["metadata"]["target"]

        def parse_and_verify_single(line: str) -> bool:
            """Verifies arithmetic, sum target, bounds, and bounds usage limits"""
            line = line.strip()
            if not line:
                return False

            # Normalize equation format: extract left side if '=' exists
            if "=" in line:
                expr_str, expr_target = line.split("=", 1)
                try:
                    if int(expr_target.strip()) != target:
                        return False
                except ValueError:
                    return False
            else:
                expr_str = line

            # Parse numbers
            tokens = [t.strip() for t in expr_str.replace("+", " ").split()]
            try:
                parsed_nums = [int(t) for t in tokens]
            except ValueError:
                return False

            # Verify target sum
            if sum(parsed_nums) != target:
                return False

            # Verify individual usage limits
            counts = {}
            for num in parsed_nums:
                if num not in max_uses_dict:
                    return False
                counts[num] = counts.get(num, 0) + 1
                if counts[num] > max_uses_dict[num]:
                    return False

            return True

        lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]

        if entry["metadata"]["mode"] == "single":
            # Any valid combination line scores full credit
            return 1.0 if any(parse_and_verify_single(line) for line in lines) else 0.0
        else:
            # Mode "all": must provide all unique valid combinations without extra/duplicate entries
            verified_solutions = set()
            for line in lines:
                if parse_and_verify_single(line):
                    # Canonical form matching formatted combinations
                    tokens = [int(t.strip()) for t in line.split("=")[0].replace("+", " ").split()]
                    canonical = self._format_combination(sorted(tokens))
                    verified_solutions.add(canonical)

            if len(verified_solutions) == len(valid_solutions) and len(lines) == len(valid_solutions):
                return 1.0
            return 0.0


class BoundedSumCombinationsCurriculum(BaseCurriculum):
    """Curriculum definition for bounded sum combinations task"""

    def __init__(self):
        super().__init__(BoundedSumCombinationsCurriculum.__name__, BoundedSumCombinationsConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="num_numbers",
                levels=list(range(3, 10)),
                default_level=0,
                description="Number of available distinct numbers",
                lower_field_name="min_numbers",
                upper_field_name="max_numbers",
            ),
            RangeAttributeDefinition(
                name="max_usage",
                levels=list(range(1, 6)),
                default_level=0,
                description="Maximum usage limit per number",
                lower_field_name="min_usage",
                upper_field_name="max_usage",
            ),
        )


# Register dataset
register_dataset(DATASET_NAME, BoundedSumCombinationsDataset, BoundedSumCombinationsConfig, BoundedSumCombinationsCurriculum)