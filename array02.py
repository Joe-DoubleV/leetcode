class Solution:
    def search(self, nums, target, leftF):
        left, right = 0, len(nums)  # [left,right]

        while left < right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1  # [left,right]
            elif nums[mid] > target:
                right = mid  # [left,right]
            else:
                # return mid                 # search only one
                if leftF:
                    right = mid  # search left
                else:
                    left = mid + 1  # search right
        return left

    def searchRange(self, nums: list, target: int) -> list:
        """34. 在排序数组中查找元素的第一个和最后一个位置
        给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
        你的算法时间复杂度必须是O(log n) 级别。
        如果数组中不存在目标值，返回[-1, -1]。"""
        if len(nums) == 0:
            return [-1, -1]
        left = self.search(nums, target, True)

        if left == len(nums) or nums[left] != target:
            return [-1, -1]

        return [left, self.search(nums, target, False) - 1]

    def searchInsert(self, nums: list, target: int) -> int:
        """35. 搜索插入位置
        给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
        如果目标值不存在于数组中，返回它将会被按顺序插入的位置。你可以假设数组中无重复元素。"""
        if not nums:
            return 0
        left, right = 0, len(nums) - 1
        mid = 0
        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        # print(left)
        return left
    """回溯算法"""
    def combinationSum(self, candidates: list, target: int):
        """39. 组合总和
        给定一个无重复元素的数组candidates和一个目标数target，
        找出candidates中所有可以使数字和为target的组合。
        candidates的数字可以无限制重复被选取。"""
        size = len(candidates)
        if size == 0:
            return []
        candidates.sort()

        def trackback(candidates:list, begin, size, path:list, res:list, residue):
            if residue == 0:
                res.append(path[:])
            for ind in range(begin,size):
                tag = residue - candidates[ind]
                if tag < 0:
                    break
                path.append(candidates[ind])
                trackback(candidates,ind,size,path,res,tag)
                path.pop()
        res = []
        trackback(candidates,0,size,[],res,target)
        # print(res)
        return res

    def combinationSum2(self, candidates: list, target: int) -> list:
        """40. 组合总和 II
        给定一个数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。
        candidates 中的每个数字在每个组合中只能使用一次,所有数字（包括目标数）都是正整数"""
        if not candidates:
            return []
        candidates.sort()
        lens = len(candidates)
        res = []
        def tackback(tag = target, path = [], begin = 0):
            if tag == 0:
                res.append(path[:])
            for ind in range(begin,lens):
                tmp = tag - candidates[ind]
                if tmp < 0 :
                    break
                if ind > begin and candidates[ind] == candidates [ind-1]:
                    continue
                tackback(tmp,path+[candidates[ind]],ind+1)
        tackback()
        # print(res)
        return res

    def rotate(self, matrix: list):
        """48. 旋转图像
        给定一个 n × n 的二维矩阵表示一个图像。将图像顺时针旋转 90 度
        你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。"""
        if not matrix or not matrix[0]:
            return
        lens = len(matrix)
        for row in range(1, lens):
            for col in range(row):
                matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]

        for row in matrix:
            for ind in range(lens//2):
                row[ind],row[lens-ind-1] = row[lens-ind-1], row[ind]
        # print(matrix)

    def maxSubArray(self, nums: list) -> int:
        """53. 最大子序和
        给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和"""

        pre, res = 0, nums[0]  # pre 表示 以 n 结尾 的 子数组最大和    #   res 所求的最大和

        for n in nums:
            pre = max(pre + n, n)  # 以 n 结尾子数组最大和 应该是 n-1（n的前一个）与n的和 与 n 的较大值
            res = max(res, pre)  # res 应该是 所有 pre 的最大值
        # print(res)
        return res

    def spiralOrder(matrix: list) -> list:
        """54. 螺旋矩阵
        给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素"""
        if not matrix or not matrix[0]:
            return []
        n = len(matrix)
        m = len(matrix[0])
        res = []
        def leftToRight():
            for ind in range(left,right+1):
                res.append(matrix[top][ind])
        def topToBottom():
            for ind in range(top,bottom):
                res.append(matrix[ind+1][right])
        def rightToLeft():
            for ind in range(right-1,left-1,-1):
                res.append(matrix[bottom][ind])
        def bottomToTop():
            for ind in range(bottom-1,top,-1):
                res.append(matrix[ind][left])
        left,right,top,bottom = 0,m-1,0,n-1
        while left <= right and top <= bottom:
            leftToRight()
            topToBottom()
            if left < right and top < bottom:
                rightToLeft()
                bottomToTop()
            left += 1
            right -= 1
            top += 1
            bottom -= 1
        # print(res)
        return res

    def canJump(self, nums: list) -> bool:
        """55. 跳跃游戏
        数组中的每个元素代表你在该位置可以跳跃的最大长度。
        判断你是否能够到达最后一个位置"""
        l = len(nums)
        rightmost = 0
        for i in range(l):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= l - 1:
                    return True
        return False

    def merge(self, intervals: list) -> list:
        """56. 合并区间
        给出一个区间的集合，请合并所有重叠的区间"""
        intervals.sort(key=lambda x: x[0])
        # print(intervals)
        res = []
        for interval in intervals:
            if not res or interval[0] > res[-1][1]:
                res.append(interval)
            else:
                res[-1][1] = max(res[-1][1], interval[1])
        return res

    def generateMatrix(self, n: int) -> list:
        """59. 螺旋矩阵 II
        给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵"""
        top, bottom, left, right = 0, n - 1, 0, n - 1
        begin = [1]
        res = [[0] * n for _ in range(n)]

        def leftToRight():
            for j in range(left, right + 1):
                res[top][j] = begin[0]
                begin[0] += 1

        def topToBottom():
            for i in range(top + 1, bottom + 1):
                res[i][right] = begin[0]
                begin[0] += 1

        def rightToLeft():
            for j in range(right - 1, left - 1, -1):
                res[bottom][j] = begin[0]
                begin[0] += 1

        def bottomToTop():
            for i in range(bottom - 1, top, -1):
                res[i][left] = begin[0]
                begin[0] += 1

        while top <= bottom:
            leftToRight()
            topToBottom()
            rightToLeft()
            bottomToTop()
            top += 1
            left += 1
            bottom -= 1
            right -= 1
        # if left == right :
        #     res[left][left] = begin[0]
        # for j in range(top+1,bottom+1):
        #     res[j][right] =
        # print(res)
        return res