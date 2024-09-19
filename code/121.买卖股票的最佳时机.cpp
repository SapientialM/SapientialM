/*
 * @lc app=leetcode.cn id=121 lang=cpp
 *
 * [121] 买卖股票的最佳时机
 */
#include <vector>
using namespace std;
// @lc code=start
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int maxProfit = 0;
        for(int i = 0; i < prices.size(); i ++){
            for(int j = i + 1; j < prices.size(); j ++){
                if( prices[j] > prices[i] ) {
                    int profit = prices[j] - prices[i];
                    if( profit > maxProfit ) maxProfit = profit;
                }
            }
        }
        return maxProfit;
    }
};
// @lc code=end

