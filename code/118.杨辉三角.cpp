/*
 * @lc app=leetcode.cn id=118 lang=cpp
 *
 * [118] 杨辉三角
 */
#include <vector>
using namespace std;
// @lc code=start
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> result;
        for(int i = 0; i < numRows; i++){
            vector<int> row = vector<int>(i);
            for(int j = 0; j < i; j++){
                if(i == 1 || i == 2){
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
// @lc code=end

