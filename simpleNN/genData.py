import random

# Generate random samples
samples = []
labels = []
for _ in range(10000):
    height = random.uniform(150, 190)
    weight = random.uniform(45, 90)
    footsize = random.uniform(34, 46)
    samples.append([height, weight, footsize])
    # Assuming a simple heuristic where 
    # weight > 60 & height > 165 and foot > 40 corresponds to male (1), 
    # otherwise female (0)
    if weight > 63 and height > 165 and footsize > 40:
        labels.append(1) # male
    else:
        labels.append(0) # female

# print("Samples:", samples[:5])  # Printing first 5 samples
# print("Labels:", labels[:5])    # Printing first 5 labels