/*
 * @lc app=leetcode.cn id=136 lang=cpp
 *
 * [136] 只出现一次的数字
 */
#include<vector>
#include<set>
using namespace std;
// @lc code=start
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
        
        return 0;
    }
};
// @lc code=end

