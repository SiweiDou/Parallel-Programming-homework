#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <vector>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

using namespace std;

// segment 表示 PCFG 中的一个“连续字符段”类型，例如 L3 表示长度为 3 的字母段，
// D2 表示长度为 2 的数字段，S1 表示长度为 1 的符号段。训练阶段统计每种长度段
// 内部具体取值的频率，生成阶段再按频率从高到低枚举候选口令。
class segment
{
public:
    // type: 字符段类别，1=字母，2=数字，3=符号；生成和训练都依赖该值选择对应表。
    int type;
    // length: 该段固定长度。同一类、同一长度的段共享一个 segment 统计对象。
    int length;
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };
    // 以 Lx/Dx/Sx 的形式输出段结构，主要用于调试和打印模型。
    void PrintSeg();
    // ordered_values: 当前 segment 下所有出现过的具体字符串，按训练频率降序排列。
    vector<string> ordered_values;
    // ordered_freqs: 与 ordered_values 一一对应的频率，用于计算候选概率。
    vector<int> ordered_freqs;
    // total_freq: 当前 segment 所有取值的频率总和，是条件概率的分母。
    int total_freq = 0;
    // values: 从具体字符串到内部编号的映射，例如 "abc" -> 0。
    unordered_map<string, int> values;
    // freqs: 从内部编号到频率的映射，与 values 配合避免重复存字符串。
    unordered_map<int, int> freqs;
    // 插入一次具体取值；若第一次出现则分配编号，否则只增加频率。
    void insert(string value);
    // 将 values/freqs 整理为按频率降序的 ordered_values/ordered_freqs。
    void order();
    // 打印当前段的所有候选取值及频率，主要用于调试训练结果。
    void PrintValues();
};

// PT(pre-terminal) 表示一个口令的结构模板，例如 L6D2S1。
// content 保存各段的类型与长度；curr_indices 表示当前模板下每段选用第几个高频取值。
class PT
{
public:
    // content: 模板中的段序列，例如 [L6, D2]。
    vector<segment> content;
    // pivot: 生成后继 PT 时的起始位置，用于避免重复生成同一组合状态。
    int pivot = 0;
    // 向模板末尾追加一个 segment。
    void insert(segment seg);
    // 打印模板结构，主要用于调试。
    void PrintPT();
    // 基于当前 curr_indices 生成下一批相邻概率状态。
    vector<PT> NewPTs();
    // curr_indices: 每个段当前选择的 ordered_values 下标，生成优先队列状态时使用。
    vector<int> curr_indices;
    // max_indices: 每个段可选取值数量的上界，防止 NewPTs 越界。
    vector<int> max_indices;
    // preterm_prob: 该结构模板本身的概率，即该 PT 在训练集中出现频率 / PT 总数。
    float preterm_prob;
    // prob: 当前具体组合状态的估计概率，等于模板概率乘以各段条件概率。
    float prob;
};

// model 保存训练得到的完整 PCFG 统计模型，包括 PT 结构频率、不同类型 segment 的
// 取值频率，以及排序后的生成入口。train/parse/order 是训练主流程。
class model
{
public:
    // 下列 id 记录各类统计对象最后分配的编号。初始为 -1，GetNext*ID 先自增再返回。
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    // 为新的 PT/字母段/数字段/符号段分配连续编号。
    int GetNextPretermID() { preterm_id++; return preterm_id; };
    int GetNextLettersID() { letters_id++; return letters_id; };
    int GetNextDigitsID() { digits_id++; return digits_id; };
    int GetNextSymbolsID() { symbols_id++; return symbols_id; };
    // total_preterm: 训练集中解析出的口令结构总数，用于计算 PT 先验概率。
    int total_preterm = 0;
    // preterminals: 所有出现过的 PT 模板。
    vector<PT> preterminals;
    // 查找一个 PT 模板是否已存在，存在返回编号，不存在返回 -1。
    int FindPT(PT pt);
    // letters/digits/symbols: 分别保存 L/D/S 三类 segment 的统计对象。
    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    // 按类型和长度查找 segment。当前模型中同一类型同一长度只保留一个统计对象。
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);
    // 各类结构/段在训练集中的出现次数，编号与上方 vector 下标一致。
    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;
    // ordered_pts: 按 PT 先验概率降序排列后的模板序列，是生成阶段优先队列的初始内容。
    vector<PT> ordered_pts;
    // 从训练集路径读取口令并更新统计模型；本期末版本在内部使用 OpenMP 局部模型归并。
    void train(string train_path);
    // 预留的模型持久化接口；当前实验主要使用 train/load 之外的在线训练流程。
    void store(string store_path);
    void load(string load_path);
    // 将单条口令切分为 L/D/S 段序列，并更新 PT、segment 和取值频率。
    void parse(string pw);
    // 训练完成后按频率排序 PT 和各 segment 的候选取值。
    void order();
    // 打印模型概要，用于调试训练结果。
    void print();
};

// PriorityQueue 是候选口令生成器。它维护按概率降序排列的 PT 状态队列，
// 每次 PopNext 取出最高概率状态生成候选，并把它的后继状态重新插入队列。
class PriorityQueue
{
public:
    // priority: 当前待展开的 PT 状态队列，保持近似概率降序。
    vector<PT> priority;
    // m: 训练得到的 PCFG 模型，生成阶段所有概率和候选值都来自该模型。
    model m;
    // 根据 PT 模板概率和各段取值频率计算当前状态概率。
    void CalProb(PT &pt);
    // 用 model::ordered_pts 初始化优先队列，并填充每个 PT 的 max_indices。
    void init();
    // 展开一个 PT 状态，生成该状态对应的一批候选口令。
    void Generate(PT pt);
    // 弹出并生成队首 PT，随后插入后继状态。
    void PopNext();
    // MPI 版本使用 do_generate 控制“只推进队列但不生成”，保证各 rank 队列状态一致。
    void PopNext(bool do_generate);
    // total_guesses: 当前 guesses 缓冲区中的候选数量。
    int total_guesses = 0;
    // old_size: Generate 前 guesses 的大小，用来记录新生成候选的写入起点。
    int old_size = 0;
    // guesses: 当前尚未被哈希处理的一批候选口令。
    vector<string> guesses;
};

// 等待 guessing.cu 中所有已提交的 GPU 生成 batch 完成，并把结果回填到 guesses。
extern void FlushGPUBatch(vector<string>& guesses);

// GPU/CPU 生成阶段的运行参数与统计信息。
struct GPUGenerateStats
{
    // gpu_items/cpu_items: 分别统计分配给 GPU 和 CPU 生成路径的候选总数。
    long long gpu_items = 0;
    long long cpu_items = 0;
    // cpu_threaded_items/cpu_serial_items: CPU 路径中 OpenMP 并行和串行生成的候选数。
    long long cpu_threaded_items = 0;
    long long cpu_serial_items = 0;
    // *_pt_count: 进入对应 CPU 路径的 PT 数量，用于判断小任务过多还是批量较均衡。
    long long cpu_threaded_pt_count = 0;
    long long cpu_serial_pt_count = 0;
    // flush_count/async_flush_count: GPU batch 提交次数，反映批量大小设置是否合适。
    long long flush_count = 0;
    long long async_flush_count = 0;
    // cached_segments: 已上传并复用的 segment 数量，用于评估 H2D 缓存效果。
    long long cached_segments = 0;
    // small/medium/large_pt_count: 动态分流策略对 PT 规模的分类统计。
    long long small_pt_count = 0;
    long long medium_pt_count = 0;
    long long large_pt_count = 0;
    // adaptive/idle/max_pt_flush_count: 触发 GPU batch flush 的不同原因计数。
    long long adaptive_flush_count = 0;
    long long idle_flush_count = 0;
    long long max_pt_flush_count = 0;
    // gpu_wait_time: CPU 等待 GPU stream 完成的累计时间；gpu_stream_time: 提交 kernel/H2D/D2H 的累计时间。
    double gpu_wait_time = 0;
    double gpu_stream_time = 0;
    // cpu_generate_time: CPU 生成候选的累计时间。
    double cpu_generate_time = 0;
    // avg_batch_items: 平均每次 GPU flush 包含的候选数。
    double avg_batch_items = 0;
    // CPU/GPU 生成阶段的估计吞吐，供实验报告分析动态分流效果。
    double cpu_items_per_sec = 0;
    double gpu_items_per_sec = 0;
    // overlap_enabled: 是否启用 CPU/GPU 生成重叠，本实现固定为 1。
    int overlap_enabled = 0;
    // gpu_threshold/batch_flush_size: 命令行传入的初始分流阈值和 batch 大小。
    int gpu_threshold = 4096;
    int batch_flush_size = 131072;
    // dynamic_gpu_threshold/adaptive_batch_target: 运行时自适应调整后的有效参数。
    int dynamic_gpu_threshold = 4096;
    int adaptive_batch_target = 131072;
};

// 配置 GPU 生成阶段的初始阈值和 batch flush 目标，由 main.cpp 解析命令行后调用。
extern void ConfigureGPUGenerate(int gpu_threshold, int batch_flush_size);
// 获取当前 rank 的 GPU/CPU 生成统计信息，由 main.cpp 汇总到全局结果。
extern GPUGenerateStats GetGPUGenerateStats();