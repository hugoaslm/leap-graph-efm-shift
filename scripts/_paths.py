import os


def find_neural_lam(module_file, cwd):
    module_file = os.path.abspath(module_file)
    base = os.path.dirname(os.path.dirname(os.path.dirname(module_file)))
    candidates = [
        os.path.join(base, "neural-lam-prob-model"),
        os.path.join(cwd, "neural-lam-prob-model"),
        os.path.join(os.path.dirname(os.path.dirname(module_file)),
                     "neural-lam-prob-model"),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "neural_lam")):
            return cand
    return candidates[0]
