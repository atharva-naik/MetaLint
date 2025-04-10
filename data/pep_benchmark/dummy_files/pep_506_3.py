import random
import string

def generate_cryptographic_hash(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation

    password = ''.join(random.sample(characters, length))
    
    return password

print(generate_cryptographic_hash())