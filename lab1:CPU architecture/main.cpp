
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <chrono> // 用于性能测试
#include <iomanip> // 用于格式化输出
using namespace std;

// 逐列访问元素
vector<double> inner_product_by_col (vector<vector<double>>& mat, vector<double>& vec){
    if (mat.empty() || vec.empty()){
        string msg = "can not calculate empty matrix or vector";
        throw invalid_argument(msg);
    }
    int r = mat.size(), c = mat[0].size(), l = vec.size();
    if (r != c || r != l){
        string msg = "the size of matrix and vector do not match";
        throw invalid_argument(msg);
    }
    vector<double> result(c, 0);
    for (int i = 0; i < c; i += 1){
        for (int j = 0; j < r; j += 1){
            result[i] += mat[j][i] * vec[j];
        }
    }
    return result;
}

// 逐行访问元素，充分利用一次读入的元素
vector<double> inner_product_cache_friendly (vector<vector<double>>& mat, vector<double>& vec){
    if (mat.empty() || vec.empty()){
        string msg = "can not calculate empty matrix or vector";
        throw invalid_argument(msg);
    }
    int r = mat.size(), c = mat[0].size(), l = vec.size();
    if (r != c || r != l){
        string msg = "the size of matrix and vector do not match";
        throw invalid_argument(msg);
    }
    vector<double> result(c, 0);
    for (int i = 0; i < r; i += 1){
        for (int j = 0; j < c; j += 1){
            result[j] += mat[i][j] * vec[i];
        }
    }
    return result;
}

// 逐个累加
double sum_ordinary(vector<double>& arr){
    double sum = 0;
    for (int i = 0; i < arr.size(); i += 1){
        sum += arr[i];
    }
    return sum;
}

// 并行友好
double sum_parallel_friendly(vector<double>& arr){
    int n = arr.size();
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    int i = 0;
    // 四路并行相加
    for (; i < n-3; i += 4){
        sum1 += arr[i];
        sum2 += arr[i+1];
        sum3 += arr[i+2];
        sum4 += arr[i+3];
    }
    // 处理不能被4整除的情况
    double result = sum1 + sum2 + sum3 + sum4;
    for (; i < n; i++) {
        result += arr[i];
    }
    return result;
}

// 真正的求和逻辑，接收一个整数序列 <0, 1, 2, ..., N-1>
template <typename Vec, size_t... Is>
double sum_impl(const Vec& arr, index_sequence<Is...>) {
    // 折叠表达式：(arr[0] + arr[1] + ... + arr[N-1])
    // 编译器会把这行代码展开成几千个加法，完全没有循环！
    return (arr[Is] + ...); 
}

// 入口函数
template <std::size_t N>
double sum_unrolled(const vector<double>& arr) {
    // 生成 0 到 N-1 的序列
    return sum_impl(arr, make_index_sequence<N>{});
}

int main() {
    // 1. 创建测试数据
    const int N = 8; // 测试矩阵大小
    vector<vector<double>> mat(N, vector<double>(N));
    vector<double> vec(N);
    
    // 填充测试数据
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i][j] = i + j + 1; // 简单的测试值
        }
        vec[i] = i + 1; // 向量值
    }
    
    // 2. 测试内积函数
    cout << "=== 测试内积函数 ===" << endl;
    try {
        // 2.1 逐列访问测试
        auto result_col = inner_product_by_col(mat, vec);
        cout << "逐列访问结果: ";
        for (double val : result_col) {
            cout << fixed << setprecision(1) << val << " ";
        }
        cout << endl;
        
        // 2.2 逐行访问测试
        auto result_cache = inner_product_cache_friendly(mat, vec);
        cout << "逐行访问结果: ";
        for (double val : result_cache) {
            cout << fixed << setprecision(1) << val << " ";
        }
        cout << endl;
        
        // 2.3 验证结果一致性
        bool consistent = true;
        for (int i = 0; i < N; i++) {
            if (abs(result_col[i] - result_cache[i]) > 1e-10) {
                consistent = false;
                break;
            }
        }
        cout << "结果一致性: " << (consistent ? "通过" : "失败") << endl;
    } catch (const exception& e) {
        cerr << "内积函数测试出错: " << e.what() << endl;
    }
    
    // 3. 测试求和函数
    cout << "\n=== 测试求和函数 ===" << endl;
    vector<double> test_arr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    
    // 3.1 普通求和
    double sum1 = sum_ordinary(test_arr);
    cout << "普通求和结果: " << sum1 << endl;
    
    // 3.2 并行友好求和
    double sum2 = sum_parallel_friendly(test_arr);
    cout << "并行友好求和结果: " << sum2 << endl;
    
    // 3.3 彻底消除循环求和
    double sum3 = sum_unrolled<8>(test_arr);
    cout << "彻底消除循环求和结果: " << sum3 << endl;
    
    // 3.4 验证结果一致性
    cout << "求和结果一致性: " 
         << (abs(sum1 - sum2) < 1e-10 && abs(sum1 - sum3) < 1e-10 ? "通过" : "失败") 
         << endl;
    
    // 4. 性能测试
    cout << "\n=== 性能测试 ===" << endl;
    const int LARGE_N = 10000; // 大规模测试
    vector<vector<double>> large_mat(LARGE_N, vector<double>(LARGE_N));
    vector<double> large_vec(LARGE_N);
    
    // 填充大规模测试数据
    for (int i = 0; i < LARGE_N; i++) {
        for (int j = 0; j < LARGE_N; j++) {
            large_mat[i][j] = i + j + 1;
        }
        large_vec[i] = i + 1;
    }
    
    // 4.1 内积函数性能测试
    auto start = chrono::high_resolution_clock::now();
    auto result_col = inner_product_by_col(large_mat, large_vec);
    auto end = chrono::high_resolution_clock::now();
    cout << "逐列访问耗时: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    start = chrono::high_resolution_clock::now();
    auto result_cache = inner_product_cache_friendly(large_mat, large_vec);
    end = chrono::high_resolution_clock::now();
    cout << "逐行访问耗时: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    
    // 4.2 求和函数性能测试
    int LARGE_N_FOR_SUM = 10000;
    vector<double> large_arr(LARGE_N_FOR_SUM);
    for (int i = 0; i < LARGE_N_FOR_SUM; i++) {
        large_arr[i] = i + 1;
    }
    volatile double g_volatile_sink = 0.0; 
    const int NUM_ITERATIONS = 1000; // 执行1000次以获取更准确的平均时间

    // 普通求和性能测试
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        g_volatile_sink = sum_ordinary(large_arr); 
    }
    end = chrono::high_resolution_clock::now();
    double avg_time_ordinary = chrono::duration_cast<chrono::microseconds>(end - start).count() / static_cast<double>(NUM_ITERATIONS);
    cout << "普通求和平均耗时: " << fixed << setprecision(2) << avg_time_ordinary << " μs" << endl;

    // 并行友好求和性能测试
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        g_volatile_sink = sum_parallel_friendly(large_arr);
    }
    end = chrono::high_resolution_clock::now();
    double avg_time_parallel = chrono::duration_cast<chrono::microseconds>(end - start).count() / static_cast<double>(NUM_ITERATIONS);
    cout << "并行友好求和平均耗时: " << fixed << setprecision(2) << avg_time_parallel << " μs" << endl;

    // 彻底消除循环求和性能测试
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sum_unrolled<LARGE_N>(large_arr);
    }
    end = chrono::high_resolution_clock::now();
    double avg_time_unrolled = chrono::duration_cast<chrono::microseconds>(end - start).count() / static_cast<double>(NUM_ITERATIONS);
    cout << "彻底消除循环求和平均耗时: " << fixed << setprecision(2) << avg_time_unrolled << " μs" << endl;
    
    return 0;
}
