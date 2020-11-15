class Solution:
    def thirdMax(self, nums: list) -> int:
        """414. 第三大的数
        给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。"""
        if not nums:
            return

        first = second = third = min(nums)
        for num in nums:
            if num <= third:
                continue
            if num > first:
                first, second, third = num, first, second
            elif first > num > second:
                second, third = num, second
            elif second > num > third:
                third = num
        if third < second < first:
            return third
        else:
            return first

    def findDuplicates(self, nums: list) -> list:
        """442. 数组中重复的数据
        给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。
        找到所有出现两次的元素。
        你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？"""
        n = len(nums)
        # for num in nums:
        #     if nums[abs(num)-1] > 0:
        #         nums[abs(num) - 1] = -nums[abs(num) - 1]
        # print(nums)
        for num in nums:
            nums[num % (n + 1) - 1] += n + 1
        return [i + 1 for i, num in enumerate(nums) if num // (n + 1) == 2]     # ==2两次，==1一次，==0没出现，==3三次

    def findDisappearedNumbers(self, nums: list) -> list:
        """448.找到所有数组中消失的数字
        给定一个范围在 1 ≤ a[i] ≤ n(n=数组大小)的整型数组，数组中的元素一些出现了两次，另一些只出现一次。
        找到所有在[1, n]范围之间没有出现在数组中的数字。
        您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。"""
        n = len(nums)
        # for num in nums:
        #     if nums[abs(num)-1] > 0:
        #         nums[abs(num) - 1] = -nums[abs(num) - 1]
        # print(nums)
        for num in nums:
            nums[num % (n + 1) - 1] += n + 1
        return [i + 1 for i, num in enumerate(nums) if num // (n + 1) == 0]     # ==2两次，==1一次，==0没出现，==3三次

    def findMaxConsecutiveOnes(self, nums: list) -> int:
        """485. 最大连续1的个数
        给定一个二进制数组， 计算其中最大连续1的个数。"""
        begin, res = -1, 0
        for i in range(len(nums)):
            if nums[i] == 1 and (i == 0 or nums[i - 1] == 0):
                begin = i
                continue
            if nums[i] == 0 and i > 0 and nums[i - 1] == 1:
                res = max(res, i - begin)
        if nums[-1] == 1:
            res = max(res, len(nums) - begin)
        return res

    def findPoisonedDuration(self, timeSeries: list, duration: int) -> int:
        """495. 提莫攻击
        在《英雄联盟》的世界中，有一个叫 “提莫” 的英雄，他的攻击可以让敌方英雄艾希（编者注：寒冰射手）进入中毒状态。
        现在，给出提莫对艾希的攻击时间序列和提莫攻击的中毒持续时间，你需要输出艾希的中毒状态总时长。
        你可以认为提莫在给定的时间点进行攻击，并立即使艾希处于中毒状态。"""
        if not timeSeries:
            return 0
        res = 0
        for i in range(1, len(timeSeries)):
            res += min(timeSeries[i] - timeSeries[i - 1], duration)
        return res + duration

    def findPairs(self, nums: List[int], k: int) -> int:
        """532.数组中的k - diff数对
        给定一个整数数组和一个整数k，你需要在数组里找到不同的k - diff数对，并返回不同的k - diff数对的数目。
        这里将k - diff数对定义为一个整数对(nums[i], nums[j])，并满足下述全部条件：
            0 <= i, j < nums.length
            i != j
            | nums[i] - nums[j] | == k
            注意， | val | 表示val的绝对值。
        """
        if not nums:
            return 0
        res = set()
        index = set()
        for num in nums:
            if num + k in index:
                res.add(num)
            if num - k in index:
                res.add(num - k)
            index.add(num)
        return len(res)

    def subarraySum(self, nums: list, k: int) -> int:
        """560. 和为K的子数组
        给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。"""
        if not nums:
            return 0
        pre, count = 0, 0
        mp = {0: 1}
        for num in nums:
            pre += num
            if pre - k in mp:
                count += mp[pre - k]

            if pre in mp:
                mp[pre] += 1
            else:
                mp[pre] = 1
        return count

    def arrayPairSum(self, nums: list) -> int:
        """561. 数组拆分 I
        给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。
        返回该 最大总和 """
        nums.sort()
        res = 0
        for i in range(0, len(nums), 2):
            res += nums[i]
        return res

    def arrayNesting(self, nums: list) -> int:
        """565. 数组嵌套
        索引从0开始长度为N的数组A，包含0到N - 1的所有整数。找到最大的集合S并返回其大小，
        其中 S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }且遵守以下的规则。
        假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，
        之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素。"""
        flag = [False] * len(nums)
        res = 0
        for i in range(len(nums)):
            if flag[i]:
                continue
            p = i
            count = 0
            while not flag[p]:
                flag[p] = True
                p = nums[p]
                count += 1

            res = max(count, res)
            if res > len(nums) // 2:
                return res
        return res

    def matrixReshape(self, nums: list, r: int, c: int) -> list:
        """566. 重塑矩阵
        在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。
        给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。
        重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。
        如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。"""
        m = len(nums)
        if m == 0:
            return nums
        n = len(nums[0])

        if m * n != r * c:
            return nums
        res = [[0] * c for _ in range(r)]
        for i in range(m * n):
            res[i // c][i % c] = nums[i // n][i % n]
        # print(res)
        return res

    def sortedSquares(A:list):
        '''给定一个按非递减顺序排序的整数数组 A，
        返回每个数字的平方组成的新数组，
        要求也按非递减顺序排序。'''
        # res = [x**2 for x in A]
        # res.sort()
        # # print(res)
        # return res
        n = len(A)
        res = [0]*n
        left,right,p = 0,n-1,n-1
        while left <= right and p>=0:
            if A[left]*A[left] < A[right]*A[right]:
                res[p] = A[right]*A[right]
                right -= 1
            else:
                res[p] = A[left]*A[left]
                left += 1
            p -= 1
        print(res)
        return res
