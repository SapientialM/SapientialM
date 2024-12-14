<<<<<<< HEAD
#include<iostream>
using namespace std;
const int N = 1e5 + 10;

int binary_search_left(int q[],int key, int l, int r){
    if(l==r){if(q[l] == key) return l; return -1;}
    int mid = (l+r) >> 1;
    if(q[mid] >= key) return binary_search_left(q, key, l, mid);
    else return binary_search_left(q, key, mid+1, r);
}

int binary_search_right(int q[],int key, int l, int r){
    if(l == r){if(q[l] == key) return l; return -1;}
    int mid = (l+r+1) >> 1;
    if(q[mid] > key) return binary_search_right(q, key, l, mid-1);
    else return binary_search_right(q, key, mid, r);
}


int main(){
    int num_1, num_2;
    cin >> num_1 >> num_2;
    int arr[num_1];
    int i = 0;
    while(i < num_1){
        cin >> arr[i];
        i ++;
    }
    //3 4
    //5 5
    //-1 -1
    int key;
    while(num_2-->0){
        cin >> key;
        cout << binary_search_left(arr, key, 0, num_1-1) << " " << binary_search_right(arr, key, 0, num_1-1) << endl;
    }


}
=======
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
>>>>>>> 5e651543976b170412ab870f2cc48509900d7b62
