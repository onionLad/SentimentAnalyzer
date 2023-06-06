import csv
import random

def generate_positive_example():
    positive_adjectives = ["great", "GREAT", "fantastic", "FANTASTIC", "amazing", "AMAZING",
                           "excellent", "EXCELLENT", "wonderful", "WONDERFUL", "very enjoyable", "really fun"]
    positive_nouns = ["movie", "concert", "museum", "restaurant", "vacation"]
    adjective = random.choice(positive_adjectives)
    noun = random.choice(positive_nouns)
    passages = [f"The {noun} was {adjective}, I loved it!",
                f"I had a {adjective} time at the {noun}! I definitely recommend going!",
                f"If you go to this {noun}, you're going to have a {adjective} time.",
                f"My kids thought the {noun} was {adjective}!"]
    return random.choice(passages)

def generate_negative_example():
    negative_adjectives = ["terrible", "TERRIBLE", "horrible", "HORRIBLE", "disappointing",
                           "awful", "AWFUL", "disastrous", "DISASTROUS", "very unenjoyable"]
    negative_nouns = ["movie", "concert", "museum", "restaurant", "vacation"]
    adjective = random.choice(negative_adjectives)
    noun = random.choice(negative_nouns)
    passages = [f"The {noun} was {adjective}, I hated it.",
                f"I had a {adjective} time at the {noun}. I don't recommend going.",
                f"If you go to this {noun}, you're going to have a {adjective} time.",
                f"My kids thought the {noun} was {adjective}."]
    return random.choice(passages)

# Generate positive and negative examples
positive_examples = [generate_positive_example() for _ in range(50)]
negative_examples = [generate_negative_example() for _ in range(50)]

# Shuffle the examples
random.shuffle(positive_examples)
random.shuffle(negative_examples)

# Combine examples and labels
examples = positive_examples + negative_examples
labels = [1] * 50 + [0] * 50

# Combine examples and labels into data
data = list(zip(examples, labels))

# Shuffle the data
random.shuffle(data)

# Write data to CSV file
with open('training_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['passage', 'label'])  # Write header
    writer.writerows(data)  # Write data rows

print("Training data generated successfully!")