#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <omp.h>

// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 * 
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 * 
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 * 
 */

// 将一个局部 segment 的统计结果合并到全局 segment 中。
// 并行训练采用“每个线程独立训练一个局部模型，最后再串行归并”的方式，
// 这样可以避免多个线程同时写 unordered_map/vector 造成数据竞争。
static void MergeSegmentStats(segment &dst, const segment &src)
{
    for (const auto &kv : src.values)
    {
        const string &value = kv.first;
        int src_id = kv.second;
        int freq = src.freqs.at(src_id);

        if (dst.values.find(value) == dst.values.end())
        {
            int new_id = dst.values.size();
            dst.values[value] = new_id;
            dst.freqs[new_id] = freq;
        }
        else
        {
            int dst_id = dst.values[value];
            dst.freqs[dst_id] += freq;
        }
    }
}

// 按 segment 类型和长度把局部模型中的 letters/digits/symbols 合并回全局模型。
// 这里复用原来的 FindLetter/FindDigit/FindSymbol 和 GetNext*ID 接口，
// 保证最终数据结构仍然符合后续 order()/Generate() 的预期。
static void MergeSegmentVector(model &global_model,
                               const vector<segment> &local_segments,
                               const unordered_map<int, int> &local_freqs,
                               int seg_type)
{
    for (int i = 0; i < local_segments.size(); i += 1)
    {
        const segment &src = local_segments[i];
        segment key(src.type, src.length);
        int dst_id = -1;

        if (seg_type == 1)
        {
            dst_id = global_model.FindLetter(key);
            if (dst_id == -1)
            {
                dst_id = global_model.GetNextLettersID();
                global_model.letters.emplace_back(key);
                global_model.letters_freq[dst_id] = 0;
            }
            global_model.letters_freq[dst_id] += local_freqs.at(i);
            MergeSegmentStats(global_model.letters[dst_id], src);
        }
        else if (seg_type == 2)
        {
            dst_id = global_model.FindDigit(key);
            if (dst_id == -1)
            {
                dst_id = global_model.GetNextDigitsID();
                global_model.digits.emplace_back(key);
                global_model.digits_freq[dst_id] = 0;
            }
            global_model.digits_freq[dst_id] += local_freqs.at(i);
            MergeSegmentStats(global_model.digits[dst_id], src);
        }
        else
        {
            dst_id = global_model.FindSymbol(key);
            if (dst_id == -1)
            {
                dst_id = global_model.GetNextSymbolsID();
                global_model.symbols.emplace_back(key);
                global_model.symbols_freq[dst_id] = 0;
            }
            global_model.symbols_freq[dst_id] += local_freqs.at(i);
            MergeSegmentStats(global_model.symbols[dst_id], src);
        }
    }
}

// 将一个线程训练出的局部 PCFG 模型归并到全局模型。
// PCFG 的统计量本质上是可加的：局部 PT 频率、segment 频率和总样本数相加后，
// 得到的模型应与串行逐行 parse 的统计结果一致。
static void MergeLocalModel(model &global_model, const model &local_model)
{
    global_model.total_preterm += local_model.total_preterm;

    MergeSegmentVector(global_model, local_model.letters, local_model.letters_freq, 1);
    MergeSegmentVector(global_model, local_model.digits, local_model.digits_freq, 2);
    MergeSegmentVector(global_model, local_model.symbols, local_model.symbols_freq, 3);

    for (int i = 0; i < local_model.preterminals.size(); i += 1)
    {
        PT pt = local_model.preterminals[i];
        int dst_id = global_model.FindPT(pt);

        if (dst_id == -1)
        {
            // 局部模型中的 PT 已经由 parse() 初始化过 curr_indices。
            // 归并时直接复制即可，不能再次追加，否则 curr_indices 长度会超过 PT 段数，
            // 后续优先队列展开 NewPTs()/Generate() 时会读到错误的状态。
            dst_id = global_model.GetNextPretermID();
            global_model.preterminals.emplace_back(pt);
            global_model.preterm_freq[dst_id] = 0;
        }

        global_model.preterm_freq[dst_id] += local_model.preterm_freq.at(i);
    }
}

// 训练的wrapper，实际上就是读取训练集
void model::train(string path)
{
    string pw;
    ifstream train_set(path);
    vector<string> passwords;
    passwords.reserve(3000000);
    int lines = 0;
    cout<<"Training..."<<endl;
    cout<<"Training phase 1: reading and parsing passwords..."<<endl;
    while (train_set >> pw)
    {
        lines += 1;
        if (lines % 10000 == 0)
        {
            // cout <<"Lines processed: "<< lines << endl;
            // 在这里更改读取的训练集口令上限
            if (lines > 3000000)
            {
                break;
            }
        }
        passwords.emplace_back(pw);
    }

    int train_threads = omp_get_max_threads();
    cout << "Training input lines: " << passwords.size() << endl;
    cout << "OpenMP train threads: " << train_threads << endl;

    vector<model> local_models(train_threads);

    // 读取单个口令之后，就可以将其扔进parse函数进行PT/segment的分割、识别、统计了
    // 并行版本中每个线程只写自己的 local_models[tid]，不共享修改全局模型，
    // 因而不需要在 parse 内部加锁，也避免了哈希表频繁加锁带来的额外开销。
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < passwords.size(); i += 1)
        {
            local_models[tid].parse(passwords[i]);
        }
    }

    cout << "Training phase 1b: merging local OpenMP models..." << endl;
    for (int i = 0; i < train_threads; i += 1)
    {
        MergeLocalModel(*this, local_models[i]);
    }
}

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(PT pt)
{
    for (int id = 0; id < preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }
        else
        {
            bool equal_flag = true;
            for (int idx = 0; idx < preterminals[id].content.size(); idx += 1)
            {
                if (preterminals[id].content[idx].type != pt.content[idx].type || preterminals[id].content[idx].length != pt.content[idx].length)
                {
                    equal_flag = false;
                    break;
                }
            }
            if (equal_flag == true)
            {
                return id;
            }
        }
    }
    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(segment seg)
{
    for (int id = 0; id < letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(segment seg)
{
    for (int id = 0; id < digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}


void segment::order()
{
    for (pair<string, int> value : values)
    {
        ordered_values.emplace_back(value.first);
    }
    // cout << "value size:" << ordered_values.size() << endl;
    std::sort(ordered_values.begin(), ordered_values.end(),
              [this](const std::string &a, const std::string &b)
              {
                  return freqs.at(values[a]) > freqs.at(values[b]);
              });

    // 将排序后的频率存入 ordered_freqs 并计算 total_freq
    for (const std::string &val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
    for (string val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }
        int id = GetNextPretermID();
        // cout << id << endl;
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        // cout << id << endl;
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT& a, const PT& b) {
    return a.preterm_prob > b.preterm_prob;  // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    bool swapped;
    cout << "total pts" << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        // cout << i << endl;
        letters[i].order();
    }
    cout << "Ordering digits" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].order();
    }
    cout << "ordering symbols" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].order();
    }
}