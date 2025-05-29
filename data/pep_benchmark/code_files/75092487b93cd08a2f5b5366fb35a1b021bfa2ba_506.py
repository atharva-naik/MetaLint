# Steven Horkey
# Summary: This is a basic command line python rpg game. Each class, method, and function has accompanying documentation.

# Importing the random module to use later to get random numbers
import random

# Defining class of Fighter for game
class Fighter:
    # init method which asserts that the fighters name is the name passed to it, and that the fighter starts with 10 points
    def __init__(self,name):
        self.name = name
        self.hit_points = 10
    # repr function method a string with the fighters name and their health points
    def __repr__(self):
        return (self.name+' (HP: '+str(self.hit_points)+')')
    # take damage method subtracts the fighters health and if they die, it prints a string that they have fallen. IF they live, it prints their name with remaning points.
    def take_damage(self,damage_amount):
        self.hit_points = self.hit_points - damage_amount
        if self.hit_points <= 0:
            print('\tAlas, '+self.name+' has fallen!')
        else:
            print('\t'+str(self.name)+' has '+str(self.hit_points)+' hit points remaining.')
    # Attack method hits or misses the other fighter depending on if a random number is greater than 12 and between 1 and 20. Sometimes the fighter misses, and the other times, the fighter hits. it then utilizes the take damage method.
    def attack(self,other):
        print(str(self.name)+' attacks '+str(other.name)+'!')
        randomNum = random.randrange(1,20)
        if randomNum >= 12:
            randHit = random.randrange(1,6)
            print('\tHits for '+str(randHit)+' hit points!')
            other.take_damage(randHit)
        else:
            print('\tMisses!')
    # the is alive method checks to see if the fighter has health over 0. IF they do, it returns true, if not, than it returns false.
    def is_alive(self):
        if self.hit_points > 0:
            return True
        else:
            return False

# combat round function is where the rpg game action happens. Based on random numbers the players strike at the same time or one before the other. It utilizes the attack and is alive methods
def combat_round(fighter1,fighter2):
    num1 = random.randrange(1,6)
    num2 = random.randrange(1,6)
    if num1 == num2:
        print('Simultaneous!')
        fighter1.attack(fighter2)
        fighter2.attack(fighter1)
    elif num1 > num2:
        fighter1.attack(fighter2)
        if fighter2.is_alive():
            fighter2.attack(fighter1)
    elif num2 > num1:
        fighter2.attack(fighter1)
        if fighter1.is_alive():
            fighter1.attack(fighter2)

# Main function creates two instances of fightesr and runs through the came utilizing the various methods and combat function. If one or both of the players die, then the battle is over and the results are printed.
def main():
    fighter1 = Fighter(input('Write your characters name '))
    fighter2 = Fighter(input('Write your opponent\'s name '))
    count = 1
    while fighter1.is_alive() and fighter2.is_alive():
        print('\n=================== ROUND '+str(count)+' ===================')
        print(fighter1.__repr__())
        print(fighter2.__repr__())
        input('Enter to Fight!')
        combat_round(fighter1,fighter2)
        count = count+1
    print('\nThe battle is over!')
    print(fighter1.__repr__())
    print(fighter2.__repr__())

# The main function is called here.
if __name__ == "__main__":
    main()

