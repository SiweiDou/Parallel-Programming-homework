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

class segment
{
public:
    int type;
    int length;
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };
    void PrintSeg();
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;
    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();
    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;
};

class model
{
public:
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID() { preterm_id++; return preterm_id; };
    int GetNextLettersID() { letters_id++; return letters_id; };
    int GetNextDigitsID() { digits_id++; return digits_id; };
    int GetNextSymbolsID() { symbols_id++; return symbols_id; };
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);
    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);
    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;
    vector<PT> ordered_pts;
    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    void CalProb(PT &pt);
    void init();
    void Generate(PT pt);
    void PopNext();
    int total_guesses = 0;
    int old_size = 0;
    vector<string> guesses;
};

// Multi-PT GPU batch flush (from guessing.cu)
extern void FlushGPUBatch(vector<string>& guesses);

// GPU/CPU 生成阶段的运行参数与统计信息。
struct GPUGenerateStats
{
    long long gpu_items = 0;
    long long cpu_items = 0;
    long long cpu_threaded_items = 0;
    long long cpu_serial_items = 0;
    long long cpu_threaded_pt_count = 0;
    long long cpu_serial_pt_count = 0;
    long long flush_count = 0;
    long long async_flush_count = 0;
    long long cached_segments = 0;
    long long small_pt_count = 0;
    long long medium_pt_count = 0;
    long long large_pt_count = 0;
    long long adaptive_flush_count = 0;
    long long idle_flush_count = 0;
    long long max_pt_flush_count = 0;
    double gpu_wait_time = 0;
    double gpu_stream_time = 0;
    double cpu_generate_time = 0;
    double avg_batch_items = 0;
    double cpu_items_per_sec = 0;
    double gpu_items_per_sec = 0;
    int overlap_enabled = 0;
    int gpu_threshold = 4096;
    int batch_flush_size = 131072;
    int dynamic_gpu_threshold = 4096;
    int adaptive_batch_target = 131072;
};

extern void ConfigureGPUGenerate(int gpu_threshold, int batch_flush_size);
extern GPUGenerateStats GetGPUGenerateStats();