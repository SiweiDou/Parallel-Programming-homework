#include "PCFG.h"
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pt_size = 0;
    if (rank == 0) {
        CalProb(pt); // 仅 rank 0 需要计算概率
        pt_size = pt.content.size();
    }
    
    // 强制所有进程同步 PT 的长度
    // 因为 rank 1~N 传入的 pt 可能是空的，如果不统一口径，会导致部分进程进 if，部分进 else 从而卡死
    MPI_Bcast(&pt_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pt_size == 1)
    {
        if (rank == 0) 
        {
            // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
            segment *a = nullptr;
            // 在模型中定位到这个segment
            if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
            if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
            if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];

            // Multi-thread TODO：
            // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
            // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
            // 这个过程是可以高度并行化的

            // 此版本MPI的多进程针对的是guess与a->ordered_values[i]的拼接，故在此不进行MPI编程，
            // 使用多线程方法进行加速
            int total_items = pt.max_indices[0];
            old_size = guesses.size();
            guesses.resize(old_size + total_items);

            if (total_items > 4000) {
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    int nth = omp_get_num_threads();
                    int thread_chunk = total_items / nth;
                    int start = tid * thread_chunk;
                    int end = (tid == nth - 1) ? total_items : (tid + 1) * thread_chunk;
                    for (int i = start; i < end; i++) {
                        guesses[old_size + i] = a->ordered_values[i];
                    }
                }
            } else {
                for (int i = 0; i < total_items; i += 1) {
                    guesses[old_size + i] = a->ordered_values[i];
                }
            }
            // int total_items = pt.max_indices[0];
            // old_size = guesses.size();
            // guesses.resize(old_size + total_items);
            
            // for (int i = 0; i < total_items; i += 1) {
            //     guesses[old_size + i] = a->ordered_values[i];
            // }
        }
    }
    else
    {
        string guess;
        int total_items = 0;
        old_size = guesses.size();
        segment *a = nullptr;

        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        if (rank == 0) {
            int seg_idx = 0;
            for (int idx : pt.curr_indices)
            {
                if (pt.content[seg_idx].type == 1) guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                if (pt.content[seg_idx].type == 2) guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                if (pt.content[seg_idx].type == 3) guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                seg_idx += 1;
                if (seg_idx == pt.content.size() - 1) break; // 留出最后一个段用于遍历展开
            }
            
            if (pt.content[pt.content.size() - 1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
            if (pt.content[pt.content.size() - 1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
            if (pt.content[pt.content.size() - 1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
            total_items = pt.max_indices[pt.content.size() - 1];// 后缀的总数
        }

        // 向所有进程广播拼接好的固定前缀guess
        int guess_len = 0;
        if (rank == 0) guess_len = guess.size();
        MPI_Bcast(&guess_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (guess_len > 0) {
            // 不使用string的指针操作，用vector<char>进行中转避免C++ 中短字符串优化(SSO)机制可能引发的内存段错误
            std::vector<char> safe_buf(guess_len);
            if (rank == 0) {
                std::copy(guess.begin(), guess.end(), safe_buf.begin());
            }
            MPI_Bcast(safe_buf.data(), guess_len, MPI_CHAR, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                guess = std::string(safe_buf.begin(), safe_buf.end());// 接收后重建为 string
            }
        }
        
        // 广播 total_items
        MPI_Bcast(&total_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (total_items > 500) 
        {
            // 计算每个进程分摊的后缀数量（向下取整）
            int strs_per_proc = total_items / size; 
            std::vector<int> local_counts(strs_per_proc);
            std::vector<char> recv_buf;

            // 主进程将后缀字符数组切割并分发
            if (rank == 0) {
                int chunk = size * strs_per_proc;
                std::vector<int> counts(chunk); // 记录每个后缀的长度
                std::vector<int> displs(chunk); // 记录每个后缀在连续内存块中的偏移量
                int total_char_len = 0;
                // 预计算偏移量并分配总内存
                for (int i = 0; i < chunk; ++i) {
                    counts[i] = a->ordered_values[i].size();
                    displs[i] = total_char_len;
                    total_char_len += counts[i];
                }

                // 将所有后缀字符串存入一维字符数组 send_buf
                std::vector<char> send_buf(total_char_len);
                int offset = 0;
                for (int i = 0; i < chunk; ++i) { 
                    const auto& s = a->ordered_values[i];
                    std::copy(s.begin(), s.end(), send_buf.begin() + offset);
                    offset += s.size();
                }

                // 统计每个进程应该接收的总字符数 (recv_counts) 及其偏移量 (recv_displs)
                std::vector<int> recv_counts(size);
                std::vector<int> recv_displs(size);
                for (int i = 0; i < size; ++i) {
                    int sum = 0;
                    for (int j = 0; j < strs_per_proc; ++j) {
                        sum += counts[i * strs_per_proc + j];
                    }
                    recv_counts[i] = sum;
                    recv_displs[i] = displs[i * strs_per_proc];
                }

                // 告诉每个进程它将收到的每个字符串的长度
                MPI_Scatter(counts.data(), strs_per_proc, MPI_INT,
                            local_counts.data(), strs_per_proc, MPI_INT,
                            0, MPI_COMM_WORLD);

                // 将一维字符数组切割并发送给对应进程
                recv_buf.resize(recv_counts[rank]);
                MPI_Scatterv(send_buf.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR,
                             recv_buf.data(), recv_counts[rank], MPI_CHAR,
                             0, MPI_COMM_WORLD);
            } else {
                // 子进程接收长度数组与字符数据
                MPI_Scatter(nullptr, strs_per_proc, MPI_INT,
                            local_counts.data(), strs_per_proc, MPI_INT,
                            0, MPI_COMM_WORLD);

                int total_len = 0;
                for (int len : local_counts) total_len += len;
                recv_buf.resize(total_len);
                MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR,
                             recv_buf.data(), total_len, MPI_CHAR,
                             0, MPI_COMM_WORLD);
            }

            // 打包所有局部结果为一条长数据流，只发一次
            if (rank == 0) {
                // Rank 0 自己先处理分配到的那一份
                guesses.resize(old_size + strs_per_proc);
                if (strs_per_proc > 500) {
                    // 提前计算好offset
                    std::vector<int> local_offsets(strs_per_proc, 0);
                    for (int i = 1; i < strs_per_proc; ++i) {
                        local_offsets[i] = local_offsets[i - 1] + local_counts[i - 1];
                    }
                    #pragma omp parallel
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        int thread_chunk = strs_per_proc / nth;
                        int start = tid * thread_chunk;
                        int end = (tid == nth - 1) ? strs_per_proc : start + thread_chunk;

                        for (int i = start; i < end; i++) {
                            int len = local_counts[i];
                            std::string substr(recv_buf.begin() + local_offsets[i], recv_buf.begin() + local_offsets[i] + len);
                            guesses[old_size + i] = guess + substr;
                        }
                    }
                } else {
                    int offset = 0;
                    for (int i = 0; i < strs_per_proc; ++i) {
                        int len = local_counts[i];
                        std::string substr(recv_buf.begin() + offset, recv_buf.begin() + offset + len);
                        guesses[old_size + i] = guess + substr;
                        offset += len;
                    }
                }
                // for (int i = 0; i < strs_per_proc; ++i) {
                //     int len = local_counts[i];
                //     std::string substr(recv_buf.begin() + offset, recv_buf.begin() + offset + len);
                //     guesses.emplace_back(guess + substr);
                //     offset += len;
                // }

                // 一次性接收其它每个进程的包裹
                for (int i = 1; i < size; ++i) {
                    int total_recv_size = 0;
                    // 接收包裹的总字节数
                    MPI_Recv(&total_recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    if (total_recv_size > 0) {
                        std::vector<char> flat_buf(total_recv_size);
                        MPI_Recv(flat_buf.data(), total_recv_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // 按 [长度(int)] + [字符本体(char[])] 的规则还原
                        // 解包数据流恢复成口令字符串
                        int ptr = 0;
                        while (ptr < total_recv_size) {
                            int rs_len = *(int*)(flat_buf.data() + ptr); // 提取口令长度
                            ptr += sizeof(int);
                            guesses.emplace_back(std::string(flat_buf.data() + ptr, rs_len)); // 提取口令本体
                            ptr += rs_len;
                        }
                    }
                }

                // 处理无法整除的残余口令
                int chunk = size * strs_per_proc;
                for (int i = chunk; i < total_items; ++i) {
                    guesses.push_back(guess + a->ordered_values[i]);
                }
            } else {
                // 非零进程将所有的结果压缩拼接到 flat 字符串流中
                std::string flat;
                if (strs_per_proc > 500) {
                    // 获取最大线程数，并为每个线程准备一个专属的字符串桶
                    int max_threads = omp_get_max_threads();
                    std::vector<std::string> thread_flats(max_threads);

                    // 解决 offset 的顺序依赖问题（计算前缀和）
                    // 提前算好每个口令在 recv_buf 中的绝对起始位置
                    std::vector<int> local_offsets(strs_per_proc, 0);
                    for (int i = 1; i < strs_per_proc; ++i) {
                        local_offsets[i] = local_offsets[i - 1] + local_counts[i - 1];
                    }

                    // 3. 开启并行域
                    #pragma omp parallel
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        int thread_chunk = strs_per_proc / nth; 
                        int start = tid * thread_chunk;
                        int end = (tid == nth - 1) ? strs_per_proc : start + thread_chunk; 

                        std::string thread_flat;
                        // 预分配内存
                        thread_flat.reserve((end - start) * guess.size());

                        for (int i = start; i < end; i++) {
                            int len = local_counts[i]; 
                            int current_offset = local_offsets[i]; // 获取当前字符串的起始位置

                            std::string substr(recv_buf.begin() + current_offset, recv_buf.begin() + current_offset + len);
                            std::string rs = guess + substr;
                            int rs_len = rs.size();
                            
                            // 将口令的长度(int)和本体(char[])塞入线程私有的缓冲区
                            thread_flat.append((char*)&rs_len, sizeof(int));
                            thread_flat.append(rs);
                        }
                        
                        // 将拼接好的私有字符串用 std::move 高效转移到对应的桶里
                        thread_flats[tid] = std::move(thread_flat);
                    } // 隐含的 barrier，等待所有线程完成

                    // 按线程顺序，把桶里的数据拼接到最终的 flat 中
                    for (int t = 0; t < max_threads; ++t) {
                        flat += thread_flats[t];
                    }
                } 
                else {
                    // 数据量小，走串行分支
                    int offset = 0;
                    for (int i = 0; i < strs_per_proc; ++i) {
                        int len = local_counts[i];
                        std::string substr(recv_buf.begin() + offset, recv_buf.begin() + offset + len);
                        std::string rs = guess + substr;
                        int rs_len = rs.size();
                        
                        flat.append((char*)&rs_len, sizeof(int));
                        flat.append(rs);
                        offset += len;
                    }
                }
                // for (int i = 0; i < strs_per_proc; ++i) {
                //     int len = local_counts[i];
                //     std::string substr(recv_buf.begin() + offset, recv_buf.begin() + offset + len);
                //     std::string rs = guess + substr;
                //     int rs_len = rs.size();
                    
                //     // 将口令的长度(int)和本体(char[])塞入同一块缓冲区
                //     flat.append((char*)&rs_len, sizeof(int));
                //     flat.append(rs);
                //     offset += len;
                // }
                
                // 将所有生成好的口令一次性发走，避开缓冲区溢出
                int total_send_size = flat.size();
                MPI_Send(&total_send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                if (total_send_size > 0) {
                    MPI_Send(flat.data(), total_send_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            if (rank == 0) {
                for (int i = 0; i < total_items; i += 1) {
                    guesses.push_back(guess + a->ordered_values[i]);
                }
            }
        }
    }
}
