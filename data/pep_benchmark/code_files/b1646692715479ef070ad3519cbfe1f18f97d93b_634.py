import random
import tkinter as tk
#NOTE: A function that determines whether the user wins or not
#      Passes the user's choice (based on what button they click)to the parameter
def get_winner(player_choice):

    # Access variables declared after the function so that the variables can be changed inside of the function
    global player_val,comp_val,output,win,win_label,output_label

    # 1. Create random integer 1-3 to use as computer's play
    comp_num = random.randint(1,3)
    # 2. Using if-statements, assign the computer to a choice (rock, paper, scissors) using the random integer generated
    if comp_num == 1:
        comp_val = "rock"
    if comp_num == 2:
        comp_val = "paper"
    if comp_num == 3:
        comp_val = "scissors"
    # 3. Determine the winner based on what the user chose and what the computer chose
    #   Rock beats Scissors
    #   Paper beats Rock
    #   Scissors beats Paper
    #   It's a tie if the computer and user chose the same object
    if comp_val == "rock":
        if player_choice == "rock":
            output = "tied"
        elif player_choice == "paper":
            output = "win"
        elif player_choice == "scissors":
            output = "lose"
    elif comp_val == "paper":
        if player_choice == "rock":
            output = "lose"
        elif player_choice == "paper":
            output = "tied"
        elif player_choice == "scissors":
            output = "win"
    elif comp_val == "scissors":
        if player_choice == "rock":
            output = "lost"
        elif player_choice == "paper":
            output = "win"
        elif player_choice == "scissors":
            output = "tied"
    # If the user wins, increase win by 1
    if output == "win":
        win +=1
    # Use the output label to write what the computer did and what the result was (win, loss, tie)
    output_label.configure(text=f"Computer did {comp_val} \n {output}")
    win_label.configure(text=f"Wins: {win}")

# Use these functions as "command" for each button
def pass_s():
    get_winner("scissors")
def pass_r():
    get_winner("rock")
def pass_p():
    get_winner("paper")

window = tk.Tk()

#Variable to count the number of wins the user gets
win = 0


#START CODING HERE
output_label =tk.Label(window,text=f"what do you choose")               ####
win_label = tk.Label(window,text=f"You haven't played yet!")



# 1. Create 3 buttons for each option (rock, paper, scissors)
greeting = tk.Label(window,text="Want to play Rock, Paper, Scissors?")

r_button= tk.Button(window,text="Rock",command=pass_r())

p_button= tk.Button(window,text="Paper",command=pass_p())

s_button= tk.Button(window,text="Scissors",command=pass_s())


# 2. Create 2 labels for the result and the number of wins



# 3. Arrange the buttons and labels using grid
greeting.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
r_button.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
p_button.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
s_button.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

output_label.pack()
win_label.pack()

window.mainloop()