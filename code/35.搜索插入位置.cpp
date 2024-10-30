/*
 * @lc app=leetcode.cn id=35 lang=cpp
 *
 * [35] 搜索插入位置
 */
#include<vector>
using namespace std;
// @lc code=start
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, r = nums.size()-1, mid = 0, ans = 0;
        while (l <= r) 
        {
            mid = (r - l) / 2 + l;
            if(nums[mid] >= target)
            {
                ans = mid;
                r =  mid - 1;
            }
            else if(nums[mid] < target)
            {
                l =  mid + 1;
            }
        }
        return ans;
        
    }
};
// @lc code=end

