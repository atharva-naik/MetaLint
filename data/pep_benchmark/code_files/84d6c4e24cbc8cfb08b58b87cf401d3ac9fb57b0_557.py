### Import two data files
Player_data = open('Player_data.csv', 'r', encoding = 'utf-8')
Match_data = open('Match_data.csv', 'r', encoding = 'utf-8')

### Make class Player
class Player:
    def __init__(self):
        self.name = ""
        self.team = ""
        self.th = 0
        self.train_dept = False
        self._8_9 = True
        self._9_10 = True
        self.count = 0
        self.injury = False

### Read data from Player_data

p = []

Player_data.readline()
for line in Player_data:
    player = Player()
    temp = line.split(',')
    player.name = temp[0]
    player.team = temp[1]
    player.th = int(temp[2])
    if temp[3] == 'O':
        player.train_dept = True
    if temp[4] == 'X':
        player._8_9 = False
    if temp[5] == 'X':
        player._9_10 = False
    player.count = int(temp[6])
    if temp[7].strip('\n') == 'O':
        player.injury = True
    p.append(player)

### Make class Match
class Match:
    def __init__(self):
        self.team = []
        self.time = ()
        self.place = ""
        self.main_ref = []
        self.assi_ref = []
        self.wait_ref = []
        self.video = []
        self.carry = []

### Read data from Match_data
m1 = Match()
m2 = Match()
m3 = Match()
m4 = Match()

m = [m1, m2, m3, m4]

round = int(input("Please enter Round: "))
for i in range(round):
    Match_data.readline()
temp = Match_data.readline().split(',')
m1.team.append(temp[1])
m1.team.append(temp[2])
m2.team.append(temp[3])
m2.team.append(temp[4])
m3.team.append(temp[5])
m3.team.append(temp[6])
m4.team.append(temp[7])
m4.team.append(temp[8].strip('\n'))

m1.time = (8,9)
m1.place = "B"
m2.time = (9,10)
m2.place = "B"
m3.time = (8,9)
m3.place = "N"
m4.time = (9,10)
m4.place = "N"

