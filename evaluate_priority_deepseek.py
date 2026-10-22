import json
from collections import Counter
from pathlib import Path

from cs329z_hw1 import adapters
from cs329z_hw1.deepseek import DeepSeekLM


# 每个分类测试多少封：
# 5 表示共测试 15 封；20 表示测试完整 60 封。
PER_CATEGORY = 20

items = json.loads(
    Path("tests/fixtures/priority_gold.json").read_text(encoding="utf-8")
)

# 保证 urgent、normal、ignore 数量相同
selected = []
selected_counts = Counter()

for item in items:
    category = item["category"]

    if selected_counts[category] < PER_CATEGORY:
        selected.append(item)
        selected_counts[category] += 1

lm = DeepSeekLM(
    max_tokens=256,
    thinking=False,
    tag=f"priority/eval-{len(selected)}",
)

correct = 0
unknown = 0
wrong = []
confusion = Counter()

for number, item in enumerate(selected, start=1):
    email = item["email"]
    expected = item["category"]
    result = adapters.run_priority(email, lm)
    actual = result["category"]

    confusion[(expected, actual)] += 1

    if actual == expected:
        correct += 1
    else:
        wrong.append(
            {
                "id": email["id"],
                "subject": email["subject"],
                "expected": expected,
                "actual": actual,
                "reason": result["reason"],
            }
        )

    if actual == "unknown":
        unknown += 1

    print(
        f"[{number:02d}/{len(selected)}] "
        f"expected={expected:<6} actual={actual:<7} "
        f"subject={email['subject'][:50]}"
    )

accuracy = correct / len(selected)

print("\n" + "=" * 70)
print(f"Correct:  {correct}/{len(selected)}")
print(f"Accuracy: {accuracy:.1%}")
print(f"Unknown:  {unknown}")

print("\nConfusion matrix:")
for (expected, actual), count in sorted(confusion.items()):
    print(f"  {expected:<6} -> {actual:<7}: {count}")

if wrong:
    print("\nWrong predictions:")
    for item in wrong:
        print("-" * 70)
        print("ID:      ", item["id"])
        print("Subject: ", item["subject"])
        print("Expected:", item["expected"])
        print("Actual:  ", item["actual"])
        print("Reason:  ", item["reason"])

print("\nUsage:")
print(lm.usage())