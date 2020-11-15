class Solution:

    def majorityElement(self, nums: list) -> int:
        """169. 多数元素
        给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
        你可以假设数组是非空的，并且给定的数组总是存在多数元素。"""
        m = len(nums)//2
        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
            if dic[num] > m:
                return num

    def rotate(self, nums: list, k: int) -> None:
        """189. 旋转数组
        给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。"""
        size = len(nums)
        k = k % size
        if size < 2:
            pass
        else:
            def reverse(left, right):
                while left < right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1

            reverse(0, size - 1)
            reverse(0, k - 1)
            reverse(k, size - 1)

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """216. 组合总和 III
        找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
        所有数字都是正整数。解集不能包含重复的组合。 """
        if n > 45 or n < 1 or k > 9 or k < 1:
            return []

        def dfs(begin, path, target):
            if len(path) == k:
                if target == 0:
                    res.append(path[:])
                return
            for i in range(begin, 10):
                if target < i:
                    break
                dfs(i + 1, path + [i], target - i)

        res = []
        dfs(1, [], n)
        return res

    def containsDuplicate(self, nums: list) -> bool:
        """217. 存在重复元素
        给定一个整数数组，判断是否存在重复元素。
        如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。"""
        numset = set(nums)

        return not len(numset) == len(nums)

    def containsNearbyDuplicate(self, nums: list, k: int) -> bool:
        """219. 存在重复元素 II
        给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，
        使得 nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k。"""
        if k < 1:
            return False
        # index = [None] * k
        n = len(nums)
        if n < 2:
            return False
        num_set = set()
        for i in range(n):
            if nums[i] in num_set:
                return True
            else:
                if i >= k:
                    num_set.remove(nums[i - k])
                num_set.add(nums[i])
        return False

    def summaryRanges(self, nums: list) -> list:
        """228. 汇总区间
        给定一个无重复元素的有序整数数组 nums 。
        返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。
        列表中的每个区间范围 [a,b] 应该按如下格式输出：
            "a->b" ，如果 a != b
            "a" ，如果 a == b
        """
        res = []
        # tmp = ""
        if not nums:
            return res
        tmp = str(nums[0])
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                continue
            else:
                if tmp != str(nums[i - 1]):
                    tmp = tmp + "->" + str(nums[i - 1])
                res.append(tmp)
                tmp = str(nums[i])

        if len(nums) > 1 and nums[-1] == nums[-2] + 1:
            tmp = tmp + "->" + str(nums[-1])
        res.append(tmp)
        return res

    def majorityElement2(self, nums: list) -> list:
        """229. 求众数 II  '摩尔投票法'
        给定一个大小为 n 的整数数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。
        进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1)的算法解决此问题。"""
        n = len(nums)
        con1, count1, con2, count2 = nums[0], 0, nums[0], 0
        for i in range(n):
            if con1 == nums[i]:
                count1 += 1
                continue
            if con2 == nums[i]:
                count2 += 1
            else:
                if count1 == 0:
                    con1 = nums[i]
                    count1 = 1
                elif count2 == 0:
                    con2 = nums[i]
                    count2 = 1
                else:
                    count1 -= 1
                    count2 -= 1
        c1, c2 = 0, 0
        for i in range(n):
            if nums[i] == con1:
                c1 += 1
                continue
            if nums[i] == con2:
                c2 += 1
        res = []
        if c1 > n // 3:
            res.append(con1)
        if c2 > n // 3:
            res.append(con2)
        return res

    def productExceptSelf(self, nums: list) -> list:
        """238. 除自身以外数组的乘积
        给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，
        其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积
        提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。
        说明: 请不要使用除法，且在O(n) 时间复杂度内完成此题。
        进阶：你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
        """
        outputs = [1] * len(nums)
        for i in range(1, len(nums)):
            outputs[i] = outputs[i - 1] * nums[i - 1]
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            outputs[i] *= right
            right *= nums[i]

        return outputs

    def missingNumber(self, nums: list) -> int:
        """268. 丢失的数字
        给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
        进阶：你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?
        nums 中的所有数字都 独一无二"""
        miss = len(nums)
        for i in range(len(nums)):
            miss += (i - nums[i])
        return miss

    def moveZeroes(self, nums: list) -> None:
        """283. 移动零
        给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
        必须在原数组上操作，不能拷贝额外的数组。
        尽量减少操作次数"""
        not_zero = -1
        for i in range(len(nums)):
            if nums[i] != 0:
                not_zero += 1
                if i != not_zero:
                    nums[i], nums[not_zero] = nums[not_zero], nums[i]

    def findDuplicate(self, nums: list) -> int:
        """287. 寻找重复数
        给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），
        可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数
            不能更改原数组（假设数组是只读的）。
            只能使用额外的 O(1) 的空间。
            时间复杂度小于 O(n2) 。
            数组中只有一个重复的数字，但它可能不止重复出现一次。"""
        slow = nums[0]
        fast = nums[nums[0]]
        while slow != fast:
            # print(slow, fast)
            slow = nums[slow]
            fast = nums[nums[fast]]
        # print(slow,fast)
        slow = 0
        while slow != fast:
            # print(slow, fast)
            slow = nums[slow]
            fast = nums[fast]
        return slow