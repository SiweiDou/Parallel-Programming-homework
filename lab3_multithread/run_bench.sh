
#!/bin/bash

# 定义要测试的可执行文件列表
EXES=(
    "main_scalar_O1"
    "main_simd_O1"
    "main_scalar_O2"
    "main_simd_O2"
)

# 定义每个版本测试的次数 (建议 3~5 次取平均值)
RUNS=3

echo "=== Benchmark Start ==="
echo "-----------------------------------------"

# 临时文件存储结果
RESULT_FILE="benchmark_result.txt"
> $RESULT_FILE

for exe in "${EXES[@]}"; do
    echo "Testing: $exe"
    
    # 检查可执行文件是否存在
    if [ ! -f "$exe" ]; then
        echo "  ERROR: $exe not found! Skipping..."
        continue
    fi

    # 循环运行多次
    for (( i=1; i<=$RUNS; i++ )); do
        echo -n "  Run $i: "
        # 运行程序，只抓取 Hash time 这一行
        ./"$exe" 2>&1 | grep "Hash time" | tee -a $RESULT_FILE
    done
    echo "-----------------------------------------"
done

echo "=== Raw Data Collected in $RESULT_FILE ==="

# 可选：简单汇总一下结果
echo ""
echo "=== Summary (Average) ==="
for exe in "${EXES[@]}"; do
    if [ -f "$exe" ]; then
        echo -n "$exe: "
        # 从结果文件中提取特定可执行文件的 Hash time，计算平均值
        grep -B 1 "$exe" "$RESULT_FILE" | grep "Hash time" | awk '{sum+=$3; count++} END {if(count>0) printf "%.4f seconds (%d runs)\n", sum/count, count}'
    fi
done
