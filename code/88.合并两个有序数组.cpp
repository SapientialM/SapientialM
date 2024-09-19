/*
 * @lc app=leetcode.cn id=88 lang=cpp
 *
 * [88] 合并两个有序数组
 */
#include <iostream>
#include <vector>
using namespace std;

// @lc code=start
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int index1 = 0, index2 = 0;
        while (index2 < n){
            if(nums1[index1] >= nums2[index2]){
                nums1.insert(nums1.begin() + index1, nums2[index2]);
                index1++;
                index2++;
            }
            index1++;
        }
    }   
};
// @lc code=end

