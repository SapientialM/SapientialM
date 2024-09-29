---
title: CCF-CSP 刷题记录
date: 2024-09-19 19:30:30
tags: [LeetCode]
categories: [算法]

---

# CCF-CSP 刷题记录

> From 2024年9月21日

# 基础算法

## 排序

### 快速排序

快排函数定义即 

- 结束条件
- 随机划分轴
- 调整轴两侧顺序
- 递归执行左右部分

```cpp
#include <iostream>
#include <vector>
#include <random>
using namespace std;

void quickSort(vector<int>& vec, int left, int right) {
    // 结束条件
    if (left >= right) 
        return;

    // 随机选择 pivot
    random_device rd;  
    default_random_engine eng(rd());
    uniform_int_distribution<int> distr(left, right);
    int pivotIndex = distr(eng);
    int pivot = vec[pivotIndex];

    // Partition
    int i = left, j = right;
    while (i <= j) {
        while (vec[i] < pivot) i++;
        while (vec[j] > pivot) j--;

        if (i <= j) {
            swap(vec[i], vec[j]);
            i++;
            j--;
        }
    }

    // 递归排序左右部分
    if (left < j)
        quickSort(vec, left, j);
    if (i < right)
        quickSort(vec, i, right);
}

int main() {
    int num = 0, temp = 0;
    vector<int> res;
    cin >> num;
    while (num-- > 0) {
        cin >> temp;
        res.push_back(temp);
    }
    quickSort(res, 0, res.size() - 1);
    for (int key : res) {
        cout << key << " ";
    }
    cout << endl;
    return 0;
}

```



### 归并排序

分治思想（n个数的排序，总共 `logn` 层，每层复杂度 `O(n)`）

- 确定分界点 `mid = (l+r) >> 1`
- 递归排序 left、right
- 有序数组归并（※） 

```cpp
#include<iostream>

using namespace std;

const int N = 1e5 + 10;

int a[N], temp[N];

void merge_sort(int q[], int l, int r){
    if(l >= r) return;
    int mid = (l+r) >> 1;
    merge_sort(q, l, mid), merge_sort(q, mid+1 , r);
    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r){
        if(q[i] <= q[j]) temp[k++] = q[i++];
        else temp[k++] = q[j++];
    }
    while(i <= mid) temp[k++] = q[i++];
    while(j <= r) temp[k++] = q[j++];
    for(i = l, j = 0; i <= r; i++, j++) q[i] = temp[j];
}

int main(){
    int n, i = 0;
    cin >> n;
    while(i < n){
        cin >> a[i];
        i++;
    }
    merge_sort(a, 0, n-1);
    for(int key : a){
        if(i == 0) break;
        i --;
        cout << key << " ";
    }
}
```

逆序对

```cpp
#include<iostream>
#include<vector>
using namespace std;

const int N = 1e5 + 10;
int a[N], temp[N];


long long int merge_sort(int vec[], int l, int r){
    
    if(l >= r) return 0;
    int mid = (l + r) >> 1;
    // 在这一步就会将 mid 左右均排好序
    long long int num = merge_sort(vec, l, mid) + merge_sort(vec, mid+1, r);
    
    int k = 0, i = l, j = mid + 1;
    
    while(i <= mid && j <= r){
        if(vec[i] > vec[j]) {
            temp[k++] = vec[j++];
            // 所以在左右排好序的情况下，mid ~ i 之间的数都是比 vec[j] 大的数，所以逆序对就是mid - i + 1
            num += mid - i + 1;
        }
        else temp[k++] = vec[i++];
    }
    
    while(i<=mid) temp[k++] = vec[i++];
    while(j<=r) temp[k++] = vec[j++];
    
    for(i=l,j=0; i <= r; i++,j++) vec[i] = temp[j];
    return num;
}

int main(){
    int n, i=0;
    cin >> n;
    while(i < n){
        cin >> a[i++];
    }
    cout << merge_sort(a,0,n-1);
    
}
```



## 二分

二分理念的一个比较重要点在于它不是狭义的要有序数列，而是只要满足左侧条件和右侧条件的递归问题就可以支撑用二分方法来解决。

- 结束条件（如 $l == r$）
- 计算 $mid$
- 递归/循环 重复更新left、right、mid 直到找到目标

二分查找数字区间

```cpp
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

    int key;
    while(num_2-->0){
        cin >> key;
        cout << binary_search_left(arr, key, 0, num_1-1) << " " << binary_search_right(arr, key, 0, num_1-1) << endl;
    }
}

```

二分用非递归也可以实现，空间损耗会更少，比如这个三次方根的题目。

```cpp
#include<iostream>  
#include<cmath>  
#include<iomanip> // 用于设置输出精度  

using namespace std;  

double pow3_f(double num){  
    return num * num * num;  
}  

double cube(double num){  
    double left, right, mid;  
    if(num < 1.0f){
        left = num;
        right = 1.0f;
    }
    else{
        left = 0;  
        right = num;
    }
    while (right - left > 1e-7) {  
        mid = (left + right) / 2.0;  
        if (pow3_f(mid) < num) {  
            left = mid;  
        } else {  
            right = mid;  
        }  
    }  
    return mid;  
}  

int main(){  
    double num;  
    cin >> num;
    if(num < 0){
        num = 0.0f - num;
        printf("-%.6f", cube(num));
    }
    else
        printf("%.6f", cube(num));
    return 0;  
}
```



