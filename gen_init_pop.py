

import random
import json

# Step 1: Function to generate a random boolean subarray of size 4
def generate_subarray():
    return [random.choice([1, 0]) for _ in range(5)]

# Step 2: Generate the main arrays
def generate_main_array():
    return [generate_subarray() for _ in range(3000)]

# 100 main arrays
arrays = [generate_main_array() for _ in range(100)]

# Step 3: Write arrays to a file
with open("population.txt", "w") as file:
    for array in arrays:
        # Use json.dumps to create a clear, readable format
        file.write(json.dumps(array))
        file.write("\n")  # newline after each array
