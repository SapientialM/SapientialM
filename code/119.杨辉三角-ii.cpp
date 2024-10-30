/*
 * @lc app=leetcode.cn id=119 lang=cpp
 *
 * [119] 杨辉三角 II
 */
#include <vector>
using namespace std;
// @lc code=start
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        vector<int> result = vector<int>();
        for(int i = 0; i <= rowIndex; i ++){
            if(i > rowIndex/2){
                result.push_back(result[rowIndex - i]);
            }
            else{
                result.push_back(pascal(rowIndex, i));
            }
        }
        return result;
    }
    int pascal(int row, int column) {
        if(row <= 1 || column == 0){
            return 1;
        }
        else{
            return pascal(row-1, column) + pascal(row-1, column-1);
        }
    }
};
// @lc code=end

