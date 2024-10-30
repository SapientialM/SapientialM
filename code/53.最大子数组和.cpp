/*
 * @lc app=leetcode.cn id=53 lang=cpp
 *
 * [53] 最大子数组和
 */
#include<vector>
using namespace std;
// @lc code=start

struct array_val
{
    /* data */
    int maxL; // 左端点
    int maxR; // 右端点
    int sum;  // L-R数组和
    int maxSum; // 最大子数组和
};

class Solution {
public:
    array_val get(vector<int>& arr, int l, int r){
        array_val result, L, R;
        if(l == r) {
            result.maxL = arr[l];
            result.maxR = arr[l];
            result.maxSum = arr[l];
            result.sum = arr[l];
            return result;
        }
        int mid = (l + r) << 1;
        L = get(arr, l, mid);
        R = get(arr, mid+1, r);
        result.maxL = max(L.maxL,L.sum + R.maxL);
        result.maxR = max(R.maxR,R.sum + L.maxR);
        int sum = 0;
        for(int i = 0; i < arr.size(); i++){
            sum += arr[i];
        }
        result.sum = sum;
        result.maxSum = max(max(result.maxL,result.maxR),(L.maxR + R.maxL));
        return result;
    }

    int maxSubArray(vector<int>& nums) {
        array_val res = get(nums, 0, nums.size() - 1);
        return res.maxSum;
    }

};
// @lc code=end


