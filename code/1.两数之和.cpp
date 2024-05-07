/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 */

#include <vector>
#include <map>
using namespace std;
// @lc code=start
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int, int> mapR;
        map<int, int>::iterator it;
        vector<int> res;
        for(int i = 0; i < nums.size(); i ++){
            if((it = mapR.find(nums[i]))!=mapR.end()){
                if(target - nums[i] == nums[i]){
                    res.push_back(i);
                    res.push_back(it->second);
                    return res;
                }
            }
            mapR[nums[i]] = i;
            if((it = mapR.find(target-nums[i]))!=mapR.end()){
                if(it->second == i){
                    continue;
                }
                res.push_back(it->second);
                res.push_back(i);
                return res;
            }
        }
        return res;
    }
};
// @lc code=end

