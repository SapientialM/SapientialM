#include <iostream>
#include <random>
using namespace std;

int main() {
    default_random_engine e(time(0)); // 随机数引擎
    uniform_int_distribution<unsigned> u(0,9);
    int num = 20;
    while (num -- > 0)
    {
        cout << u(e) << endl;
    }
    
    
    

    return 0;
}