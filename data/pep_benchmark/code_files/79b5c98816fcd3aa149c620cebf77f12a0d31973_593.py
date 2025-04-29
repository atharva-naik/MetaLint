#Submitted by thr3sh0ld
#Logic: DP. dp[i][j] denotes the minimum health require to move to the grid. health should be positive
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        r, c = len(dungeon), len(dungeon[0])
        dp = [[0]*(c+1) for _ in range(r+1)]
        dp[r-1][c-1] = 1
        #two below loops help to store the minimum health to move to next room 
        for i in range(r-2, -1, -1):
            dp[i][c-1] = max(1, dp[i+1][c-1]-dungeon[i+1][c-1])
        for i in range(c-2, -1, -1):
            dp[r-1][i] = max(1, dp[r-1][i+1]-dungeon[r-1][i+1])
        for i in range(r-2, -1, -1):
            for j in range(c-2, -1, -1):
                right = max(1, dp[i][j+1]-dungeon[i][j+1])
                down = max(1, dp[i+1][j]-dungeon[i+1][j])
                dp[i][j] = min(right, down)
        return max(1, dp[0][0]-dungeon[0][0])