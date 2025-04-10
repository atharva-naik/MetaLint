import random
import string

def generate_password(length=12):
    """function for generating password using the random module"""
    characters = string.ascii_letters + string.digits + string.punctuation

    password =  "".join(random.choice(characters) for x in range(16))
    
    return password

print(generate_password())