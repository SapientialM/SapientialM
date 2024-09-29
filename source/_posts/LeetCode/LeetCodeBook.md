---
title: 算法笔记
date: 2024-09-19 19:30:30
tags: [LeetCode, algorithm]
categories: [算法笔记]
---

# 算法笔记

## 1 [labuladong的算法笔记](https://labuladong.online/algo/home/)

> From 2024年9月20日

### **单链表的基本技巧**

1、[合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists) `Easy`

> 虚拟头节点，双指针

2、[单链表分解](https://leetcode.cn/problems/partition-list) `middle`

> 虚拟头节点，双指针

3、[合并K个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists) `hard`

> 堆的使用，priority_queue

4、寻找单链表的倒数第 `k` 个节点

5、寻找单链表的中点

6、判断单链表是否包含环并找出环起点

7、判断两个单链表是否相交并找出交点



## 2 LeeCode 刷题笔记

> From 2024年9月20日 待补充

### [455. 分发饼干](https://leetcode.cn/problems/assign-cookies/)

> 09/22 [提交](https://leetcode.cn/submissions/detail/566906740/)

### [岛屿的周长](https://leetcode.cn/problems/island-perimeter/)

> 09/23 [提交](https://leetcode.cn/submissions/detail/567257105/)

### [最大连续 1 的个数](https://leetcode.cn/problems/max-consecutive-ones/)

> 09/24 [提交](https://leetcode.cn/submissions/detail/567534692/)

### [提莫攻击](https://leetcode.cn/problems/teemo-attacking/)

> 09/25 [提交](https://leetcode.cn/submissions/detail/567912182/)

### [下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/)

> 09/26 [提交](https://leetcode.cn/submissions/detail/568196305/)

### [键盘行](https://leetcode.cn/problems/keyboard-row/)

> 09/27 [提交](https://leetcode.cn/submissions/detail/568435534/)

### [相对名次](https://leetcode.cn/problems/relative-ranks/)

> 09/28 [提交](https://leetcode.cn/submissions/detail/568764757/)



## C++ 自查表

> 所有容器：[Containers library - cppreference.com](https://en.cppreference.com/w/cpp/container#Container_adaptors)
>
> 中文首页：[cppreference.com](https://zh.cppreference.com/w/首页)
>
> 英文首页：[cppreference.com](https://en.cppreference.com/w/)
>
> 如果以下英文网页难以理解，可以将 url 中的 en 改为 zh，即中文网页。

### 算法 algorithm

### 随机数

## C++中的默认随机数引擎（default_random_engine）

在C++中，`default_random_engine`是一个生成伪随机数的引擎类，它至少提供了相对随意、非专家或轻量级使用的可接受的引擎行为。这个引擎类是标准库实现中的选择，用于生成伪随机数。

**代码示例**

使用`default_random_engine`可以非常简单地生成随机数。以下是一个基本的使用示例：

```cpp
#include <random>
#include <iostream>
using namespace std;

int main() {

    default_random_engine e; // 随机数引擎

    for (int i = 0; i < 10; ++i)

    cout << e() << endl; // 返回一个无符号整数

    return 0;

}
```

此代码创建了一个`default_random_engine`对象，并在循环中调用它来生成随机数。`e()`调用不接受参数，并返回一个无符号整数。

**生成指定范围内的随机数**

要生成特定范围内的随机数，可以使用随机数分布类，如`uniform_int_distribution`。以下是一个生成0到9之间随机数的示例：

```cpp
int main() {

    uniform_int_distribution<unsigned> u(0,9); // 随机数分布类

    default_random_engine e; // 随机数引擎

    for (int i = 0; i < 10; ++i)

    cout << u(e) << endl; // 注意是u(e)，而不是u(e())

    return 0;

}
```

这段代码首先定义了一个`uniform_int_distribution`对象`u`，它的范围是0到9。然后，它创建了一个`default_random_engine`对象`e`，并在循环中调用`u(e)`来生成随机数。

**设置随机数引擎种子**

为了每次运行程序时生成不同的随机数序列，可以设置随机数引擎的种子。通常使用时间作为种子：

```cpp
uniform_int_distribution<int> u(0, 9);

default_random_engine e(time(0)); // 使用当前时间作为种子

for (int i = 0; i < 10; ++i)

cout << u(e) << " ";
```

这段代码使用当前时间作为种子来初始化`default_random_engine`对象`e`，然后生成随机数。

**注意事项**

- 对于给定的发生器，每次运行返回相同的数值序列。如果希望在程序的每次运行中都生成不同的序列，需要设置不同的种子。
- 如果在定义时没有设置种子，`default_random_engine`将使用默认种子。
- 如果在循环中使用时间作为种子，由于时间的单位是秒，可能会导致在短时间内生成相同的种子，从而产生相同的随机数序列。

通过使用`default_random_engine`和适当的分布类对象，可以更直观地获取随机数，而不需要进行取余等计算。这是C++11中引入的一种新的获取随机数的方法，相比于传统的`rand`函数，它提供了更好的随机性和灵活性。

#### 约束 （Since C++20）

包含：[Constrained algorithms (since C++20) - cppreference.com](https://en.cppreference.com/w/cpp/algorithm/ranges)

- Non-modifying sequence operations
- Modifying sequence operations
- Partitioning operations
- Sorting operations
- Binary search operations (on sorted ranges)
- Set operations (on sorted ranges)
- Heap operations
- Minimum/maximum operations
- Permutation operations
- etc.....

### 顺序容器

#### \<vector\> 

**头文件**：`#include<vector>`

**描述**：向量容器，类似于数组，存储在连续内存块，可以在末尾快速插入与删除，支持随机访问。

**函数**：[std::vector - cppreference.com](https://en.cppreference.com/w/cpp/container/vector)



#### \<string\>

**头文件**：`#include<string>`

**描述**：字符串容器，除开字符串操作外，还包含序列容器操作。

**函数**：[std::basic_string - cppreference.com](https://en.cppreference.com/w/cpp/string/basic_string)



#### \<deque>

**头文件**：`#include<deque>`

**描述**：双端队列容器，块中地址连续，块间地址不连续，从首尾快速插入与删除，支持随机访问。

> 分配空间比vector速度块，因为重新分配空间后原有元素不需要复制

**函数**：[std::deque - cppreference.com](https://en.cppreference.com/w/cpp/container/deque)



#### \<list>

**头文件**：`#include<list>`

**描述**：链表容器，双链表类模板，需从表头开始遍历。

**函数**：[std::list - cppreference.com](https://en.cppreference.com/w/cpp/container/list)



#### \<forward_list>

**头文件**：`#include<forward_list>`

**描述**：链表容器，单链表类模板，需从表头开始遍历，与list相比有更低的存储成本。

**函数**：[std::forward_list - cppreference.com](https://en.cppreference.com/w/cpp/container/forward_list)



### 关联容器

| [set](https://en.cppreference.com/w/cpp/container/set)       | collection of unique keys, sorted by keys (class template)   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [map](https://en.cppreference.com/w/cpp/container/map)       | collection of key-value pairs, sorted by keys, keys are unique (class template) |
| [multiset](https://en.cppreference.com/w/cpp/container/multiset) | collection of keys, sorted by keys (class template)          |
| [multimap](https://en.cppreference.com/w/cpp/container/multimap) | collection of key-value pairs, sorted by keys (class template) |

>multi 即支持元素重复，map 即 key 映射 value，set 即 仅包含 value。

### 无序关联容器

| [unordered_set](https://en.cppreference.com/w/cpp/container/unordered_set)(C++11) | collection of unique keys, hashed by keys (class template)   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map)(C++11) | collection of key-value pairs, hashed by keys, keys are unique (class template) |
| [unordered_multiset](https://en.cppreference.com/w/cpp/container/unordered_multiset)(C++11) | collection of keys, hashed by keys (class template)          |
| [unordered_multimap](https://en.cppreference.com/w/cpp/container/unordered_multimap)(C++11) | collection of key-value pairs, hashed by keys (class template) |

> 无序关联容器基于哈希实现，平均复杂度 O(1)，最坏 O(n)。

### 适配器容器

| [stack](https://en.cppreference.com/w/cpp/container/stack)   | adapts a container to provide stack (LIFO data structure) (class template) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [queue](https://en.cppreference.com/w/cpp/container/queue)   | adapts a container to provide queue (FIFO data structure) (class template) |
| [priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue) | adapts a container to provide priority queue (class template) |
| [flat_set](https://en.cppreference.com/w/cpp/container/flat_set)(C++23) | adapts a container to provide a collection of unique keys, sorted by keys (class template) |
| [flat_map](https://en.cppreference.com/w/cpp/container/flat_map)(C++23) | adapts two containers to provide a collection of key-value pairs, sorted by unique keys (class template) |
| [flat_multiset](https://en.cppreference.com/w/cpp/container/flat_multiset)(C++23) | adapts a container to provide a collection of keys, sorted by keys (class template) |
| [flat_multimap](https://en.cppreference.com/w/cpp/container/flat_multimap)(C++23) | adapts two containers to provide a collection of key-value pairs, sorted by keys (class template) |

#### \<queue\>

**头文件**：`#include<queue>`

**描述**：队列容器，FIFO特点，不允许顺序遍历。

**函数**：[std::queue - cppreference.com](https://en.cppreference.com/w/cpp/container/queue)

- empty(): 判断容器是否空
- size(): 返回实际元素个数
- frot(): 返回队头元素
- back(): 返回队尾元素
- push(elem): elem 进队
- pop(): 元素出队



#### \<priority_queue\>

```cpp
template<
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type>
> class priority_queue;
```

**头文件**：`#include<priority_queue>`

**描述**：优先队列容器，任意顺序进入队列，出队操作将以出队优先级为准（默认最大元素出队）。

**函数**：[std::priority_queue - cppreference.com](https://en.cppreference.com/w/cpp/container/priority_queue)

- empty()
- size()
- push(elem)
- top(): 返回队头元素
- pop(): 元素出队

> `< >` 指定模板参数。默认情况下，`std::priority_queue`使用`std::less`作为其比较对象，这意味着队列会按照元素的升序排列（即，最大的元素被视为最高优先级），通常`std::priority_queue`默认是最大堆。

> 这里有一个很难理解的地方，使用Compare时，会以Compare为True作为最高优先级，But 堆的内部实现是vector，每次出堆时实际上是**堆顶元素和最后一个元素互换**，我们可以理解为**将其沉入数组末端**。那么实际上假如没有出堆这一个过程而是继续将数据缓存在vector中，所有元素出堆后，形成的数组就是升序的。只是由于**出堆这一个操作让我们看起来是一个反序的排列**，而实际上元素出堆在这里相当于**逆序输出**。
>
> 所以这里的最高优先级，可以理解为，**高优先级的放在前列**，但是实际上每次**出堆都会弹出尾部**，所以**Compare变现为逆序**。

## Lambda 表达式（since C++11）

[Lambda expressions (since C++11) - cppreference.com](https://en.cppreference.com/w/cpp/language/lambda)

### 非泛型Lambda表达式

#### 基本语法

没有显式模板参数列表的Lambda表达式的基本语法如下：

```cpp
[capture](parameters) mutable noexcept(expression) -> return_type {  
    // 函数体  
}
```

- **capture**：捕获列表，指定Lambda表达式体内部可以访问的外部变量。捕获可以是值捕获（通过`=`或`&`指定）或引用捕获（仅通过`&`指定）。
- **parameters**：参数列表，与普通函数相同，但在这个上下文中是可选的。
- **mutable**、**noexcept**、**-> return_type**：这些都是可选的。`mutable`允许在Lambda表达式体内修改捕获的变量（如果它们是值捕获的）。`noexcept`指定Lambda表达式是否抛出异常。`-> return_type`指定Lambda表达式的返回类型，如果Lambda表达式体中有返回语句且编译器无法从返回语句推断出返回类型，则需要显式指定。

#### 例子

```cpp
#include <iostream>  
#include <vector>  
#include <algorithm>  
  
int main() {  
    std::vector<int> vec = {1, 2, 3, 4, 5};  
  
    // 使用非泛型Lambda表达式对vector中的每个元素加1  
    std::transform(vec.begin(), vec.end(), vec.begin(),  
        [](int x) { return x + 1; });  
  
    for (int n : vec) {  
        std::cout << n << ' ';  
    }  
    // 输出: 2 3 4 5 6  
  
    return 0;  
}
```

比如这个例子，Lambda表达式`[](int x) { return x + 1; }`是一个非泛型Lambda表达式，在这里的作用等同于：

```cpp
bool cmp(int x){
    return x + 1;
}
std::transform(vec.begin(), vec.end(), vec.begin(),cmp);  
```

#### 作用

非泛型Lambda表达式在C++中不能直接用来“定义”一个具有全局或命名空间作用域的函数，因为Lambda表达式本质上是匿名函数对象，它们通常在需要函数对象、回调函数或类似机制的地方使用。

但你可以将Lambda表达式赋值给一个**函数指针**（Lambda [capture list]为空其参数类型和返回类型与函数指针兼容），或者赋值给一个`std::function`对象。

如：

- **赋值函数指针**

```cpp
#include <iostream>  
#include <functional>  
  
int main() {  
    // 使用std::function来存储Lambda表达式  
    std::function<int(int, int)> add = [](int x, int y) { return x + y; };  
    // 也可以用auto来自动推导Lambda表达式 Since C++11
  	auto add = [](int x, int y) { return x + y; }; 
    // 调用Lambda表达式  
    std::cout << "The sum is: " << add(2, 3) << std::endl;  
  
    return 0;  
}
```

- **作为参数传递函数**

```cpp
#include <iostream>  
#include <vector>  
#include <algorithm>  
  
void printVector(const std::vector<int>& vec, std::function<void(int)> printer) {  
    for (int n : vec) {  
        printer(n);  
    }  
}  
  
int main() {  
    std::vector<int> vec = {1, 2, 3, 4, 5};  
  
    // 将Lambda表达式作为参数传递给printVector  
    printVector(vec, [](int n) { std::cout << n << ' '; });  
  
    return 0;  
}
```

