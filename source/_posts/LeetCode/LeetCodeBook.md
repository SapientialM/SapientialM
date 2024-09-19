---
title: LeetCodeCookBook 刷题记录
date: 2024-06-13 18:09:38
tags: [LeetCode]
categories: [算法]
# STL参考：https://wyqz.top/p/870124582.html
# 刷题参考：https://books.halfrost.com/leetcode/ChapterTwo/Array/
---



# C++ STL 刷题手册

> 作者的bb：一个月没动静了，赶紧活动起来啊！！！

有些简单介绍也不重复，参考[cplusplus.com](https://cplusplus.com/reference/stl/)就可以了，我在这里就是记录一些笔记，方便刷题。

## 1. vector
- size 和 capacity 的区别，如何动态删除元素、释放内存？
- 指针和引用的作用是什么？
- shrink_to_fit 和 resize 有什么作用？
- 如何截取片段？
## 2. stack

## 3. queue

## 4. deque

## 5. priority_queue

## 6. map
K-V 键值对
## 7. set
K 集合

```cpp
s.begin()					//返回指向第一个元素的迭代器
s.end()						//返回指向最后一个元素的迭代器
s.clear()					//清除所有元素
s.count()					//返回某个值元素的个数
s.empty()					//如果集合为空，返回true，否则返回false
s.equal_range()				//返回集合中与给定值相等的上下限的两个迭代器
s.erase()					//删除集合中的元素
s.find(k)					//返回一个指向被查找到元素的迭代器
s.insert()					//在集合中插入元素
s.lower_bound(k)			//返回一个迭代器，指向键值大于等于k的第一个元素
s.upper_bound(k)			//返回一个迭代器，指向键值大于k的第一个元素
s.max_size()				//返回集合能容纳的元素的最大限值
s.rbegin()					//返回指向集合中最后一个元素的反向迭代器
s.rend()					//返回指向集合中第一个元素的反向迭代器
s.size()					//集合中元素的数目

```
## 8. pair

## 9. string

## 10. bitset

## 11. array

## 12. tuple

---



# 刷题记录

该刷题记录主要是记录自己的日常刷题，给自己提供一些手感和积累，题目来源于LeetCode，希望通过日积月累，能有所提升和感触。

题单参考的是 [halfrost/LeetCode-Go](https://books.halfrost.com/leetcode/)，希望大家都能一起加油。

主要流程的话，目前计划的是分三轮：

- 第一轮刷所有简单难度且AC>40%的题目，当然第一轮的时候就可以开始整理题型了

> 有一个跟着我进度的题单，当然肯定很慢hhh，[题单陆续更新ing](https://leetcode.cn/problem-list/GCJ4Hsn9/)

- 第二轮刷所有中等难度且AC>40%的题目
- 第三轮查漏补缺，关注自己薄弱的方向

## A. 数组 Array
### [1.两数之和](https://leetcode.cn/problems/two-sum/)

> 2024/04/22

很简单的一道题，最简单的方法是暴搜，双指针搜索，和为 target 记录下来，复杂度$O(n^2)$：
```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        for(int i = 0; i < nums.size() - 1; i ++)
            for(int j = i + 1; j < nums.size(); j ++)
                if(nums[i] + nums[j] == target)
                {
                    res.push_back(i);
                    res.push_back(j);
                    return res;
                }
        return res;
    }
};
```
尝试一下将复杂度降低，用哈希表，单指针遍历，寻找差值，复杂度$O(n)$：
```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end()) {
                return {it->second, i};
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};
```

### [26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

> 2024/06/13

很简单的一道题，也是双指针的一个理念，两个值 pos 和 i ，pos 负责赋值不重复元素，i 负责往后遍历，复杂度$O(n)$：

```C++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int pos = 1;
        for(int i = 1; i < nums.size(); i++) {
            if(nums[i] > nums[i-1]){
                nums[pos] = nums[i];
                pos++;
            }
        }
        return pos;
    }
};
```

做到这里的时候，思考到了 vector 长度的问题，这道题返回了 pos ，所以不需要作长度的处理。但是实际上如果不返回 pos 要修改 nums 长度的话得用 resize，但是实际上 resize 更改的是逻辑大小，shrink_to_fit 在 C++ 11 引入，可以把物理大小和逻辑大小匹配起来。同时在这里也不得不想到 vector 的属性 size 和 capacity，具体的话去看刷题手册，这里就不细说了。



### [27. 移除元素](https://leetcode.cn/problems/remove-element/)

> 2024/06/19 （看算法去了，每日一题下次一定）

同样简单的题，双指针，类似于上面的 26 题，一个指针覆盖数组，一个指针遍历数组。k 在 ++ 之前其实就可以当第一个指针，所以就直接用 k 作为指针了。

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int k = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            if(nums[i] != val)
            {
                nums[k] = nums[i];
                k++;
            }
        }
        return k;
    }
};
```



### [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

> 2024/06/25

经典的错误，标准的零分 bushi。$O(logn)$复杂度，有序数组查找，所以二分查找。

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, r = nums.size()-1, mid = 0, ans = nums.size();
        while (l <= r) 
        {
            mid = (r + l) / 2;
            if(nums[mid] >= target)
            {
                ans = mid;
                r =  mid - 1;
            }
            else
            {
                l =  mid + 1;
            }
        }
        return ans;
        
    }
};
```



###  [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

> 2024/08/01

分治

```cpp
class Solution {
public:
    struct array_val
    {
        /* data */
        int maxL; // 左端点
        int maxR; // 右端点
        int sum;  // L-R数组和
        int maxSum; // 最大子数组和
    };

    array_val get(vector<int>& arr, int l, int r){
        array_val result, L, R;
        if(l == r) {
            result.maxL = arr[l];
            result.maxR = arr[l];
            result.maxSum = arr[l];
            result.sum = arr[l];
            return result;
        }
        int mid = (l + r) >> 1;
        L = get(arr, l, mid);
        R = get(arr, mid + 1, r);
        result.maxL = max(L.maxL,L.sum + R.maxL);
        result.maxR = max(R.maxR,R.sum + L.maxR);
        int sum = 0;
        for(int i = l; i <= r; i++){
            sum += arr[i];
        }
        result.sum = sum;
        result.maxSum = max(max(L.maxSum,R.maxSum),(L.maxR + R.maxL));
        return result;
    }
    int maxSubArray(vector<int>& nums) {
        array_val res = get(nums, 0, nums.size() - 1);
        return res.maxSum;
    }
};
```



### [66. 加一](https://leetcode.cn/problems/plus-one/)

> 2024/08/02

数学

```cpp
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        for(int i = digits.size()-1; i >= 0; i --){
            digits[i] ++;
            digits[i] = digits[i] % 10;
            if(digits[i] != 0) return digits;
        }
        digits.insert(digits.begin(), 1);
        return digits;
        
    }
};
```



### [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

> 2024/08/04

双指针

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int index1 = 0, index2 = 0, jump = 0;
        while (index2 < n){
            if(nums1[index1] >= nums2[index2]){
                nums1.insert(nums1.begin() + index1, nums2[index2]);
                index2++;
                index1++;
                continue;
            }
            if(jump >= m){
                nums1.insert(nums1.begin() + index1, nums2[index2]);
                index2++;
            }
            index1++;
            jump++;
        }
        nums1 = std::vector<int>(nums1.begin(), nums1.begin() + n + m);
    }
};
```

### [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

> 2024/08/06

DFS、BST

```cpp
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return dfs(nums, 0, nums.size() - 1);
    }

    TreeNode* dfs(vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }

        int mid = (left + right) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = dfs(nums, left, mid - 1);
        root->right = dfs(nums, mid + 1, right);
        return root;
    }
};
```

### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

> 2024/09/03

数学

```cpp
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> result;
        for(int i = 0; i < numRows; i++){
            vector<int> row = vector<int>(0);
            for(int j = 0; j <= i; j++){
                if(i == 0 || i == 1 || j == 0 || j == i){
                    row.push_back(1);
                }
                else{
                    row.push_back(result[i-1][j] + result[i-1][j-1]);
                }
            }
            result.push_back(row);
        }
        return result;
    }
};
```



### [119. 杨辉三角 II](https://leetcode.cn/problems/pascals-triangle-ii/)

> 2024/09/03

数学

```cpp
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        vector<vector<int>> result(rowIndex + 1);
        for (int i = 0; i <= rowIndex; ++i) {
            result[i].resize(i + 1);
            result[i][0] = result[i][i] = 1;
            for (int j = 1; j < i; ++j) {
                result[i][j] = result[i - 1][j - 1] + result[i - 1][j];
            }
        }
        return result[rowIndex];
    }
};
```



### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

> 2024/09/04

遍历、数学

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int INF = 1e9;
        int minprice = INF, maxprofit = 0;
        for (int price: prices) {
            maxprofit = max(maxprofit, price - minprice);
            minprice = min(price, minprice);
        }
        return maxprofit;
    }
};
```



### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

> 2024/09/05

哈希、位运算

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        set<int> number_set;
        set<int>::iterator it;
        for(int i = 0; i < nums.size(); i ++){
            if((it = number_set.find(nums[i])) != number_set.end()){
                number_set.erase(it);
            }
            else{
                number_set.insert(nums[i]);
            }
        }
        for( int value : number_set){
            return value;
        }
        return 0;
    }
};
```

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ret = 0;
        for (auto e: nums) ret ^= e;
        return ret;
    }
};
```






## B. 字符串 String



## C. 双指针 Two Pointers



## D. 链表 Linked List



## E. 栈 Stack



## F. 树 Tree



## G. 动态规划 Dynamic Programming



## H. 回溯算法 Backtracking



## I. 深度优先搜索 Depth First Search



## J. 宽度优先搜索 Breadth First Search



## K. 二分查找 Binary Search



## L. 数论 Math



## M. 哈希表 Hash Table
### [205.同构字符串](https://leetcode.cn/problems/isomorphic-strings/description/)

> 2024/04/17

同样简单的一道题，最容易想到的是两个哈希表相互映射进行比较，复杂度$O(n)$，如下：
```c++
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        unordered_map<char, char>::iterator iterS;
        unordered_map<char, char>::iterator iterT;
        unordered_map<char, char> mapS;
        unordered_map<char, char> mapT;
        for(int i = 0; i < s.size(); i ++){
            iterS = mapS.find(s[i]);
            if(iterS != mapS.end()){
                iterT = mapT.find(t[i]);
                if(iterT == mapT.end()){
                    return false;
                }
                if(!(iterS->second == t[i] && iterT->second == s[i])){
                    return false;
                }
            }
            else{
                iterT = mapT.find(t[i]);
                if(iterT != mapT.end()){
                    return false;
                }
                mapS[s[i]] = t[i];
                mapT[t[i]] = s[i];
            }
        }
        return true;
    }
};
```


## 排序 Sort



## 位运算 Bit Manipulation



## 联合查找 Union Find



## 滑动窗口 Sliding Window



## 线段树 Segment Tree



## 二叉索引树 Binary Indexed Tree



# 刷题笔记

分治法：





# 工具

[Markdown 简介 | 开源文档 (osdoc.net)](https://osdoc.net/md/)

# 参考

[LeetCode-Go](https://github.com/halfrost/LeetCode-Go)

[代码随想录](https://www.programmercarl.com/)

《挑战程序设计竞赛2：算法与数据结构》

[《挑战程序设计竞赛》习题册攻略 ](https://github.com/yogykwan/acm-challenge-workbook)





