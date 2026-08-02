import os, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_paths
import test_prepare

MODULES = [test_paths, test_prepare]


def main():
    failures = []
    total = 0
    for mod in MODULES:
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            total += 1
            fn = getattr(mod, name)
            try:
                fn()
                print(f"PASS {mod.__name__}.{name}")
            except Exception:
                failures.append(f"{mod.__name__}.{name}")
                print(f"FAIL {mod.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{total - len(failures)}/{total} passed")
    if failures:
        print("Failed:", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
