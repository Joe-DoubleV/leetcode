class Solution:
    def twoSum(self, nums, target: int) :
        """1. 两数之和
        给定一个整数数组 nums 和一个目标值 target，
        请你在该数组中找出和为目标值的那 两个 整数，
        并返回他们的数组下标。"""

        '''在进行迭代并将元素插入到表中的同时，
        我们还会回过头来检查表中是否已经存在当前元素所对应的目标元素。
        如果它存在，那我们已经找到了对应解，并立即将其返回'''
        tmp = {}
        for ind in range(len(nums)):
            if target - nums[ind] in tmp:
                return [tmp[target - nums[ind]],ind]
            tmp[nums[ind]] = ind

    def maxArea(self, height: list):
        """11. 盛最多水的容器
        给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点(i,ai) 。
        在坐标内画 n 条垂直线，垂直线 i的两个端点分别为(i,ai) 和 (i, 0)。
        找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。"""

        res = 0
        left, right = 0, len(height) - 1
        while left < right:
            if height[left] < height[right]:
                res = max(res,height[left]*(right-left))
                left += 1
            else:
                res = max(res,height[right]*(right-left))
                right -= 1
        return res

    def threeSum(self, nums: list) -> list:
        """15. 三数之和
        给你一个包含 n 个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，
        使得a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
        注意：答案中不可以包含重复的三元组。"""
        size = len(nums)
        if size == 0:
            return []
        nums.sort()
        # print(nums)
        res = []
        for index in range(size):

            if index == 0 or nums[index] != nums[index - 1]:
                left, right = index + 1, size - 1
                while left < right:
                    # if left > index+1 and nums[left] == nums[left-1]:
                    #     left += 1
                    #     continue
                    # if right < size - 1 and nums[right] == nums[right+1]:
                    #     right -= 1
                    #     continue
                    sums = nums[index] + nums[left] + nums[right]
                    if sums > 0 or (right < size - 1 and nums[right] == nums[right + 1]):
                        right -= 1
                    elif sums < 0 or (left > index + 1 and nums[left] == nums[left - 1]):
                        left += 1
                    elif sums == 0:
                        res.append([nums[index], nums[left], nums[right]])
                        left += 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        right -= 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
        # print(res)
        return res

    def threeSumClosest(self, nums: list, target: int) -> int:
        """16. 最接近的三数之和
        给定一个包括n 个整数的数组nums和 一个目标值target。找出nums中的三个整数，使得它们的和与target最接近。
        返回这三个数的和。假定每组输入只存在唯一答案。"""
        if not nums:
            return None
        nums.sort()
        res = sum(nums[0:3])

        def closest(num):
            nonlocal res
            if abs(num - target) < abs(res - target):
                res = num

        l = len(nums)
        for ind in range(l):
            if ind > 0 and nums[ind] == nums[ind - 1]:
                continue
            left, right = ind + 1, l - 1
            while left < right:
                sums = nums[ind] + nums[left] + nums[right]
                if sums == target:
                    return target
                elif sums > target:
                    right -= 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                    closest(sums)
                else:
                    left += 1
                    while left < right and nums[left - 1] == nums[left]:
                        left += 1
                    closest(sums)
        # print(res)
        return res

    def fourSum(self, nums: list, target: int) -> list:
        """18. 四数之和
        给定一个包含n 个整数的数组nums和一个目标值target，判断nums中是否存在四个元素 a，b，c和 d，
        使得a + b + c + d的值与target相等？找出所有满足条件且不重复的四元组。"""

        size = len(nums)
        nums.sort()
        res = []
        for first in range(size - 3):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            for second in range(first + 1, size - 2):
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                four = size - 1
                for third in range(second + 1, size - 1):
                    if third > second + 1 and nums[third] == nums[third - 1]:
                        continue

                    while third < four and nums[second] + nums[third] + nums[four] > target - nums[first]:
                        four -= 1
                    if third == four:
                        break
                    if nums[first] + nums[second] + nums[third] + nums[four] == target:
                        res.append([nums[first], nums[second], nums[third], nums[four]])
        # print(res)
        return res

    def removeDuplicates(self, nums: list) -> int:
        """26. 删除排序数组中的重复项
        给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成"""
        size = len(nums)
        if size < 2:
            return size
        real, temp = 0, 1
        while temp < size:
            if nums[temp] == nums[real]:
                temp += 1
                continue
            else:
                real += 1
                nums[real] = nums[temp]
                temp += 1
        # print(nums,real+1)
        return real + 1

    def removeElement(self, nums: list, val: int) -> int:
        """27. 移除元素
        给你一个数组 nums和一个值 val，你需要 原地 移除所有数值等于val的元素，并返回移除后数组的新长度。
        不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
        元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素"""
        if not nums:
            return 0
        size = len(nums)

        left, right = 0, size - 1
        while left < right:
            if nums[left] == val:
                while left < right and nums[right] == val:
                    right -= 1
                if left == right:
                    return left
                else:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1
            else:
                left += 1
        # print(nums)
        if nums[left] == val:
            return left
        else:
            return left + 1

    def search(self, nums: list, target: int) -> int:
        """33. 搜索旋转排序数组
        给你一个整数数组 nums ，和一个整数 target 。
        该整数数组原本是按升序排列，但输入时在预先未知的某个点上进行了旋转。（例如，数组[0,1,2,4,5,6,7]可能变为[4,5,6,7,0,1,2] ）。
        请你在数组中搜索target ，如果数组中存在这个目标值，则返回它的索引，否则返回-1
        """
        left, right = 0, len(nums) - 1
        if right == -1:
            return -1
        while left < right:
            mid = (left + right) // 2
            # print(left,right,mid)
            if nums[mid] < target:
                if nums[left] <= nums[mid]:  # left sored
                    left = mid + 1  # [mid, right]
                else:  # right sored
                    if nums[right] < target:
                        right = mid
                    else:
                        left = mid + 1
            else:
                if nums[left] <= nums[mid]:  # left sored
                    if nums[left] <= target:
                        right = mid
                    else:
                        left = mid + 1
                else:  # right sored
                    right = mid
        return left if nums[left] == target else -1