from threading import Timer
from .Animal_Book import *
from .AuctionSlot import *
from .AuctionSlot1 import *
from .AuctionSlot2 import *
from .Auctionslot3 import *
import random

class Auction:


    max_auction_number = 3
    currAuctionNumber = 0
    auctionPrice = 0
    AIList = []
    time = 0
    currentPrice = 0
    auctionAnimal = 0
    bid_player = 0


    def addAI(self,ai):
        self.AIList.append(ai)

    def bid(self, AIList): #AI bidding
        return 0


    def create_auction(self,auctionNumber):
        print("creating auction")
        if self.currAuctionNumber < self.max_auction_number:
            rand_num = random.randint(0,len(AnimalBook.animalarray)-1)
            rand_animal = AnimalBook.animalList[AnimalBook.animalkey[rand_num]]
            self.currAuctionNumber += 1
            if auctionNumber == 1:
                print("create a1")
                return AuctionSlot1(auctionNumber,rand_animal)
            elif auctionNumber ==2:
                return AuctionSlot2(auctionNumber,rand_animal)
            elif auctionNumber ==3:
                return AuctionSlot3(auctionNumber,rand_animal)


