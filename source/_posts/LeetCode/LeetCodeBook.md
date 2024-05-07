---
title: LeetCodeCookBook 刷题记录
date: 2024-04-23 16:55:38
tags: [LeetCode]
---

# 算法

## 数组 Array
### 1. [1.两数之和](https://leetcode.cn/problems/two-sum/)
很简单的一道题，最简单的方法是暴力解决，复杂度$O(n^2)$，如下：
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
但是做题都想来个最优解，所以尝试一下将复杂度降低。用哈希表：
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

## 字符串 String



## 双指针 Two Pointers



## 链表 Linked List



## 栈 Stack



## 树 Tree



## 动态规划 Dynamic Programming



## 回溯算法 Backtracking



## 深度优先搜索 Depth First Search



## 宽度优先搜索 Breadth First Search



## 二分查找 Binary Search



## 数论 Math



## 哈希表 Hash Table
### 1.[205.同构字符串](https://leetcode.cn/problems/isomorphic-strings/description/)
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

# 动图工具
FlipaClip

# 参考

[LeetCode-Go](https://github.com/halfrost/LeetCode-Go)

[代码随想录](https://www.programmercarl.com/)

《挑战程序设计竞赛2：算法与数据结构》

[《挑战程序设计竞赛》习题册攻略 ](https://github.com/yogykwan/acm-challenge-workbook)



