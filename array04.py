class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """80. 删除排序数组中的重复项 II
        给定一个增序排列数组 nums ，你需要在 原地 删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成"""
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

    def merge(self, nums1: list, m: int, nums2: list, n: int) -> None:
        """88. 合并两个有序数组
        给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
        初始化nums1 和 nums2 的元素数量分别为m 和 n 。
        你可以假设nums1有足够的空间（空间大小大于或等于m + n）来保存 nums2 中的元素"""
        p1, p2 = m - 1, n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p1 + p2 + 1] = nums1[p1]
                p1 -= 1
            else:
                nums1[p1 + p2 + 1] = nums2[p2]
                p2 -= 1
        if p1 < 0:
            nums1[0:p2 + 1] = nums2[0:p2 + 1]

    def subsetsWithDup(self, nums: list) -> list:
        """90. 子集 II
        给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
        说明：解集不能包含重复的子集。"""
        size = len(nums)

        def _dfs(begin, path):
            if depth == len(path):
                res.append(path)
                return
            for i in range(begin, size):
                if i > begin and nums[i] == nums[i - 1]:
                    continue
                _dfs(i + 1, path + [nums[i]])

        path = []
        res = []
        nums.sort()
        for depth in range(size + 1):
            _dfs(0, path)
        return res

    def buildTree(self, preorder: list, inorder: list) -> TreeNode:
        """105. 从前序与中序遍历序列构造二叉树
        根据一棵树的前序遍历与中序遍历构造二叉树。
        你可以假设树中没有重复的元素。"""
        def _buildTree(pre_left, pre_right, in_left, in_right):
            if pre_left > pre_right:
                return None
            # 前序遍历第一个节点就是root
            pre_root = pre_left
            in_root = inorder.index(preorder[pre_root])
            size_left_sub = in_root - in_left
            # root
            root = TreeNode(preorder[pre_root])
            # left
            root.left = _buildTree(pre_left + 1, pre_left + size_left_sub, in_left, in_root - 1)
            # right
            root.right = _buildTree(pre_left + size_left_sub + 1, pre_right, in_root + 1, in_right)
            return root

        n = len(preorder)
        return _buildTree(0, n - 1, 0, n - 1)

    def buildTree2(self, inorder: list, postorder: list) -> TreeNode:
        """106. 从后序与中序遍历序列构造二叉树
        根据一棵树的后序遍历与中序遍历构造二叉树。
        你可以假设树中没有重复的元素。"""
        def _buildTree(in_left,in_right,post_left,post_right):
            if post_left > post_right:
                return None
            # 后序遍历最后一个节点就是root
            post_root = post_right
            in_root = inorder.index(postorder[post_root])
            size_left_sub = in_root - in_left
            # root
            root = TreeNode(postorder[post_root])
            # left
            root.left = _buildTree(in_left,in_root-1,post_left,post_left+size_left_sub-1)
            # right
            root.right = _buildTree(in_root+1,in_right,post_left+size_left_sub,post_right-1)
            return root
        n = len(inorder)
        return _buildTree(0,n-1,0,n-1)

    def generate(self, numRows: int) -> list:
        """118. 杨辉三角
        给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。"""
        if numRows == 0:
            return []
        res = [[1]]
        for x in range(1, numRows):
            tmp = [1] * (x + 1)

            for i in range(1, x):
                tmp[i] = res[-1][i - 1] + res[-1][i]
            res.append(tmp)
        return res

    def getRow(self, rowIndex: int) -> list:
        """119. 杨辉三角 II
        给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。"""
        res = [1]*(rowIndex+1)
        if rowIndex >= 2:
            for i in range(3,rowIndex+2):
                for j in range(i-2,0,-1):
                    res[j] += res[j-1]
        return res

    def minimumTotal(self, triangle: list) -> int:
        """120. 三角形最小路径和
        给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
        相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。"""
        n = len(triangle)
        dp = [0] * n
        for i in range(n):
            for j in range(i, -1, -1):
                if j == i:
                    dp[j] = dp[j - 1] + triangle[i][j]
                elif j == 0:
                    dp[j] = dp[j] + triangle[i][j]
                else:
                    dp[j] = min(dp[j], dp[j - 1]) + triangle[i][j]
        # print(dp)
        return (min(dp))

    def maxProfit(self, prices: list) -> int:
        """121. 买卖股票的最佳时机
        给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
        如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
        注意：你不能在买入股票前卖出股票。"""
        n = len(prices)
        if n == 0:
            return 0
        dp = [0] * n
        minPrice = prices[0]
        for i in range(n):
            minPrice = min(minPrice, prices[i])
            dp[i] = max(dp[i - 1], prices[i] - minPrice)
        # print(dp)
        return dp[-1]

    def maxProfit2(self, prices: list) -> int:
        """122. 买卖股票的最佳时机 II
        给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
        注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。"""
        if not prices: return 0
        n = len(prices)
        f0 = - prices[0]
        f1 = 0
        for i in range(1, n):
            tmp0 = max(f0, f1 - prices[i])
            tmp1 = max(f1, f0 + prices[i])
            f0, f1 = tmp0, tmp1
        return f1

    def maxProduct(self, nums: list) -> int:
        """152. 乘积最大子数组
        给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。"""
        fmin, fmax = 1, 1
        ans = nums[0]
        for num in nums:
            fmax, fmin = max(fmax * num, fmin * num, num), min(fmax * num, fmin * num, num)
            ans = max(ans, fmax)
        return ans

    def twoSum(self, numbers: list, target: int) -> list:
        """167. 两数之和 II - 输入有序数组
        给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
        函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
        返回的下标值（index1 和 index2）不是从零开始的。
        你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。"""
        left, right = 0, len(numbers)-1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left+1, right+1]
            elif numbers[left] + numbers[right] < target:
                left += 1
            else:
                right -= 1
        return [-1, -1]