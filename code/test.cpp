#include <iostream>
#include <vector>
#include <set>
using namespace std;
int singleNumber(vector<int>& nums) {
    set<int> number_set;
    set<int>::iterator it;
    for(int i = 0; i < nums.size(); i ++){
        if((it = number_set.find(nums[i])) != number_set.end()){
            cout << "find: " << nums[i] << endl;
            number_set.erase(it);
        }
        else{
            cout << "insert: " << nums[i] << endl;
            number_set.insert(nums[i]);
        }
    }
    for (auto value : number_set) {
        cout << "return: " << value << endl;
        return value;
    }
    
    return 0;
}


int main(int argc, char *argv[]) {
    vector<int> numbers = {2,2,1};
    singleNumber(numbers);
}
