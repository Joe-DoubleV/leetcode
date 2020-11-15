class Solution:

    def uniquePaths(self, m: int, n: int) -> int:
        """62. 不同路径
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
        问总共有多少条不同的路径。"""

        if m <= 0 or n <= 0:
            return 0
        if m == 1 or n == 1:
            return 1
        res = [[1] * m for _ in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                res[i][j] = res[i][j - 1] + res[i - 1][j]
        # print(res)
        return res[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: list) -> int:
        """63. 不同路径
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
        问总共有多少条不同的路径。现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径"""
        if not obstacleGrid or not obstacleGrid[0] or obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
            return 0
        n = len(obstacleGrid)
        m = len(obstacleGrid[0])
        res = [[0] * m for _ in range(n)]
        for i in range(n):

            for j in range(m):
                if i == 0 and j == 0:
                    res[i][j] = 1
                elif i == 0 and obstacleGrid[i][j] == 0:
                    res[i][j] = res[i][j - 1]
                elif j == 0 and obstacleGrid[i][j] == 0:
                    res[i][j] = res[i - 1][j]
                elif obstacleGrid[i][j] == 0:
                    res[i][j] = res[i - 1][j] + res[i][j - 1]
        return res[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """64. 最小路径和
        给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
        说明：每次只能向下或者向右移动一步"""
        n = len(grid)
        m = len(grid[0])
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j - 1]
                elif j == 0:
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        # print(grid)
        return grid[-1][-1]

    def plusOne(self, digits: list) -> list:
        """66.加一
        给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
        最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
        你可以假设除了整数 0 之外，这个整数不会以零开头"""
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                return digits
            else:
                digits[i] = 0
                if i == 0:
                    digits.insert(0, 1)
        return digits

    def setZeroes(self, matrix: list) -> None:
        """73.矩阵置零
        给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。
        """
        if matrix and matrix[0]:
            n, m = len(matrix), len(matrix[0])
            zerosRow = set()
            zerosCol = set()
            for i in range(n):
                for j in range(m):
                    if matrix[i][j] == 0:
                        zerosRow.add(i)
                        zerosCol.add(j)

            for i in range(n):
                for j in range(m):
                    if i in zerosRow or j in zerosCol:
                        matrix[i][j] = 0

    def searchMatrix(self, matrix: list, target: int) -> bool:
        """74.搜索二维数组
        编写一个高效的算法来判断m x n矩阵中，是否存在一个目标值。该矩阵具有如下特性：
        每行中的整数从左到右按升序排列。每行的第一个整数大于前一行的最后一个整数。"""
        n = len(matrix)
        if n == 0:
            return False
        # m = len(matrix[0])

        up, down = 0, n - 1
        while up < down:
            mid = (up + down) >> 1
            if matrix[mid][-1] < target:
                up = mid + 1
            else:
                down = mid
        return target in matrix[up]

    def sortColors(self, nums: list) -> None:
        """75.颜色分类
        给定一个包含红色、白色和蓝色，一共n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
        此题中，我们使用整数 0、1 和 2 分别表示红色、白色和蓝色。"""
        left, right = 0, len(nums) - 1
        curr = 0
        while curr <= right:
            if nums[curr] == 2:
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
            elif nums[curr] == 0:
                nums[curr], nums[left] = nums[left], nums[curr]
                left += 1
                curr += 1
            else:
                curr += 1

    def subsets(self, nums: list) -> list:
        """78.子集
        给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）"""
        size = len(nums)
        if size == 0:
            return [[]]

        def _dfs(begin, path):
            if depth == len(path):
                res.append(path[:])
                return
            for i in range(begin, size):
                # if not path or i > path[-1]:
                path.append(nums[i])
                _dfs(i + 1, path)
                path.pop()

        path = []
        res = []
        for depth in range(size + 1):
            _dfs(0, path)
        return res

    def exist(self, board: list, word: str) -> bool:
        """79.单词搜索
        给定一个二维网格和一个单词，找出该单词是否存在于网格中。单词必须按照字母顺序，通过相邻的单元格内的字母构成，
        其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。"""
        directions = {(0, 1), (0, -1), (1, 0), (-1, 0)}
        n = len(board)
        if n == 0:
            return False
        m = len(board[0])
        marked = [[False] * m for _ in range(n)]

        # print(marked)
        def _find(index, start_x, start_y):
            if index == len(word) - 1:
                return board[start_x][start_y] == word[-1]

            if board[start_x][start_y] == word[index]:
                marked[start_x][start_y] = True
                for direction in directions:
                    new_x = start_x + direction[0]
                    new_y = start_y + direction[1]
                    if 0 <= new_x < n and 0 <= new_y < m and not marked[new_x][new_y] and _find(index + 1, new_x, new_y):
                        return True
                marked[start_x][start_y] = False
            return False

        for i in range(n):
            for j in range(m):
                if _find(0, i, j):
                    return True
        return False

    def removeDuplicates(self, nums: list) -> int:
        """80. 删除排序数组中的重复项 II
        给定一个增序排列数组 nums ，你需要在 原地 删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。"""
        nex, count = 1, 1
        for font in range(1, len(nums)):
            if nums[font] == nums[font - 1]:
                count += 1
            else:
                count = 1
            if count <= 2:
                nums[nex] = nums[font]
                nex += 1
        return nex