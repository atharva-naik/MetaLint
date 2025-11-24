import os
import json
import matplotlib.pyplot as plt

# main
if __name__ == "__main__":
    error_dist = {
        "adjacent": 6,
        "before/after error": 2,
        "definition vs initialization": 2,
        "plausible looking": 23,
        "random": 14,
        "repetition": 3,   
    }
    print(sum(error_dist.values()))