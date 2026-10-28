"""Write the arithmetic prompt set the Monarch e2e runs train on (no dataset download needed).

    python tests/manual/monarch_e2e/make_dataset.py ~/miles_e2e/data/arith.jsonl
"""

import json
import random
import sys
from pathlib import Path


def main(path: str, num_prompts: int = 2000, seed: int = 0) -> None:
    rng = random.Random(seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _ in range(num_prompts):
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            prompt = f"What is {a} + {b}? Answer briefly and put the final answer in \\boxed{{}}."
            f.write(json.dumps({"prompt": [{"role": "user", "content": prompt}], "label": str(a + b)}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
