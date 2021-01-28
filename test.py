# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
#new_fellowship = [____ for ____ in fellowship ____]
new_fellowship = [member if len(member)>=7 else "" for member in fellowship]
# Print the new list
print(new_fellowship)