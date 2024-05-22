import csv
import random

# read tsv file
with open('../data/WikiQA.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    dataset = list(reader)

questions = []

for data in dataset:
    question = data[1]
    if question not in questions:
        questions.append(question)

# write the questions to a text file
with open('../data/WikiQA-questions.txt', 'w', encoding='utf-8') as f:
    for question in questions:
        f.write(question + '\n')

# select a random sample of 50 questions
random.seed(0)
sample = random.sample(questions, 50)

# write the sample to a text file
with open('../data/WikiQA-questions-sample.txt', 'w', encoding='utf-8') as f:
    for question in sample:
        f.write(question + '\n')
