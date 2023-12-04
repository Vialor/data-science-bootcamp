# 1. Display Fibonacci Series upto 10 terms
fib1, fib2 = 0, 1
for i in range(10):
  print(fib2)
  fib1, fib2 = fib2, fib1 + fib2

# 2. Display numbers at the odd indices of a list
l = [1, 2, 3, 4, 5]
for i in range(0, len(l), 2):
  print(l[i])

# 3. Print a list in reverse order
print(l[::-1])

# 4. Your task is to count the number of different words in this text
from collections import Counter
import re


string = """ChatGPT has created this text to provide tips on creating interesting paragraphs. 
	First, start with a clear topic sentence that introduces the main idea. 
	Then, support the topic sentence with specific details, examples, and evidence.
	Vary the sentence length and structure to keep the reader engaged.
	Finally, end with a strong concluding sentence that summarizes the main points.
	Remember, practice makes perfect!
	"""
wordList = re.split(" |\n\t", string)
wordList = filter(lambda word: word, wordList)
wordList = [word[:-1] if word[-1] in ",.!" else word for word in wordList]
print(len(Counter(wordList).keys()))


# 5. Write a function that takes a word as an argument and returns the number of vowels in the word
def countVowls(word: str):
  count = 0
  for c in word:
    if c in "aeiou":
      count += 1
  return count

# 6. Iterate through the following list of animals and print each one in all caps.
animals = ['tiger', 'elephant', 'monkey', 'zebra', 'panther']
for animal in animals:
  print(animal.upper())

# 7. Iterate from 1 to 15, printing whether the number is odd or even
for i in range(1, 16):
  print("even" if i % 2 == 0 else "odd")

# 8. Take two integers as input from user and return the sum
def stdinSum():
  a = input("first int: ")
  b = input("second int: ")
  return int(a) + int(b)
print(stdinSum())