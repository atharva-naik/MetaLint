import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer


class WorkerRushBot(sc2.BotAI):
    # on_step function is called for every game step
    # it takes current game state and iteration
    async def on_step(self, iteration):
        # if it is the first frame on the game
        # if iteration == 0:
        for worker in self.workers:
            # have the worker attack the enemy start position
            await self.do(worker.attack(self.enemy_start_locations[0]))


run_game(maps.get("(2)16-BitLE"), [
    Bot(Race.Protoss, WorkerRushBot()),
    Computer(Race.Protoss, Difficulty.VeryEasy)
], realtime=True)
