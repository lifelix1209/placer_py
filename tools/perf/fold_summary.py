"""Self and inclusive sample share per function from py-spy's folded stacks."""
import sys
from collections import Counter

self_n, incl_n, total = Counter(), Counter(), 0
with open(sys.argv[1]) as handle:
    for line in handle:
        stack, _, count = line.rstrip("\n").rpartition(" ")
        n = int(count)
        total += n
        frames = [f.split(" (")[0] + " " + f.split(" (")[1].split(":")[0].split("/")[-1]
                  if " (" in f else f for f in stack.split(";")]
        self_n[frames[-1]] += n
        for f in set(frames):
            incl_n[f] += n
print(f"samples {total}")
print("== inclusive")
for f, n in incl_n.most_common(45):
    print(f"{100 * n / total:6.1f}%  {f}")
print("== self")
for f, n in self_n.most_common(30):
    print(f"{100 * n / total:6.1f}%  {f}")
