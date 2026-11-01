#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// Convert character to integer index for C and Occ tables
// Alphabet mapping: '$' -> 0, 'A' -> 1, 'C' -> 2, 'G' -> 3, 'T' -> 4
inline int char_to_idx(char c) {
    switch (c) {
        case '$': return 0;
        case 'A': case 'a': return 1;
        case 'C': case 'c': return 2;
        case 'G': case 'g': return 3;
        case 'T': case 't': return 4;
        default: return -1;
    }
}

inline char idx_to_char(int idx) {
    const char alphabet[] = "$ACGT";
    if (idx >= 0 && idx < 5) return alphabet[idx];
    return '?';
}

// Read FASTA file and return cleaned DNA sequence string
std::string read_fasta(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return "";
    }
    std::string sequence = "";
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '>') continue; // Skip header and empty lines
        for (char c : line) {
            c = std::toupper(c);
            if (c == 'A' || c == 'C' || c == 'G' || c == 'T') {
                sequence += c;
            }
        }
    }
    file.close();
    return sequence;
}

// Structure representing the FM-Index
struct FMIndex {
    std::string bwt;
    std::vector<int> SA;                 // Suffix Array
    std::vector<int> C;                  // Cumulative character count table
    std::vector<std::vector<int>> Occ;   // Rank/Occurrence matrix (5 x N+1)

    FMIndex() : C(6, 0) {}
};

// Build Suffix Array, BWT, C-Table, and Occ-Matrix
FMIndex build_fm_index(const std::string& text) {
    FMIndex index;
    int n = text.size();

    // 1. Build Suffix Array
    index.SA.resize(n);
    std::iota(index.SA.begin(), index.SA.end(), 0);

    // Sort suffix indices lexicographically
    std::sort(index.SA.begin(), index.SA.end(), [&text](int i, int j) {
        return text.compare(i, std::string::npos, text, j, std::string::npos) < 0;
    });

    // 2. Build BWT String (Parallelized with OpenMP if enabled)
    index.bwt.resize(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (index.SA[i] == 0) {
            index.bwt[i] = '$';
        } else {
            index.bwt[i] = text[index.SA[i] - 1];
        }
    }

    // 3. Build C-Table (Cumulative counts for $, A, C, G, T)
    std::vector<int> counts(5, 0);
    for (char c : text) {
        int idx = char_to_idx(c);
        if (idx != -1) counts[idx]++;
    }
    index.C[0] = 0;
    for (int i = 1; i <= 5; ++i) {
        index.C[i] = index.C[i - 1] + counts[i - 1];
    }

    // 4. Build Occ-Table (Rank matrix)
    index.Occ.assign(5, std::vector<int>(n + 1, 0));
    for (int i = 0; i < n; ++i) {
        #pragma omp simd
        for (int c = 0; c < 5; ++c) {
            index.Occ[c][i + 1] = index.Occ[c][i];
        }
        int char_idx = char_to_idx(index.bwt[i]);
        if (char_idx != -1) {
            index.Occ[char_idx][i + 1]++;
        }
    }

    return index;
}

// Backward Search query matching with verbose step printing option
std::pair<int, int> backward_search(const FMIndex& index, const std::string& pattern, bool debug = false) {
    int m = pattern.size();
    int c_idx = char_to_idx(pattern[m - 1]);
    if (c_idx == -1) return {-1, -1};

    int L = index.C[c_idx];
    int R = index.C[c_idx + 1] - 1;

    if (debug) {
        std::cout << "   [Step 1] Initial range for suffix '" << pattern[m - 1] 
                  << "': L=" << L << ", R=" << R << " (Count=" << (R - L + 1) << ")\n";
    }

    for (int i = m - 2; i >= 0 && L <= R; --i) {
        c_idx = char_to_idx(pattern[i]);
        if (c_idx == -1) return {-1, -1};

        L = index.C[c_idx] + index.Occ[c_idx][L];
        R = index.C[c_idx] + index.Occ[c_idx][R + 1] - 1;

        if (debug) {
            std::cout << "   [Step " << (m - 1 - i) << "] Update range for suffix '" 
                      << pattern.substr(i) << "': L=" << L << ", R=" << R 
                      << " (Count=" << (R >= L ? (R - L + 1) : 0) << ")\n";
        }
    }

    if (L > R) return {-1, -1};
    return {L, R};
}

// Naive pattern search for verification ground truth
int naive_search_count(const std::string& text, const std::string& pattern) {
    int count = 0;
    size_t pos = text.find(pattern, 0);
    while (pos != std::string::npos) {
        count++;
        pos = text.find(pattern, pos + 1);
    }
    return count;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "   BWT & FM-Index Pattern Searcher (Demo & Verification)\n";
    std::cout << "========================================================\n";

    #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        std::cout << "OpenMP Status: ENABLED (Threads: " << num_threads << ")\n";
    #else
        std::cout << "OpenMP Status: DISABLED (Running in sequential mode)\n";
    #endif

    std::string fasta_path = (argc > 1) ? argv[1] : "GCF_000005845.2_ASM584v2_genomic.fna";
    std::cout << "Loading FASTA file: " << fasta_path << "...\n";
    
    std::string sequence = read_fasta(fasta_path);
    if (sequence.empty()) {
        std::cout << "Notice: Could not load '" << fasta_path << "'. Falling back to internal test sequence.\n";
        sequence = "AGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
    }

    std::string text = sequence + "$";
    std::cout << "Loaded Genome Length: " << sequence.length() << " bp\n";

    // 1. Build FM-Index
    auto start_build = std::chrono::high_resolution_clock::now();
    FMIndex index = build_fm_index(text);
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();

    std::cout << "FM-Index Construction Time: " << build_time << " ms\n";
    std::cout << "\n--- C-Table (Cumulative Character Base Offsets) ---\n";
    std::cout << "  $: " << index.C[0] << " | A: " << index.C[1] 
              << " | C: " << index.C[2] << " | G: " << index.C[3] 
              << " | T: " << index.C[4] << " | Total: " << index.C[5] << "\n";

    // 2. Educational Step-by-Step Backward Search Walkthrough
    std::cout << "\n--- Step-by-Step Backward Search Walkthrough for 'ATCG' ---\n";
    backward_search(index, "ATCG", true);

    // 3. Verification of standard queries against brute force ground truth
    std::cout << "\n--- Functional Correctness Verification ---\n";
    std::vector<std::string> queries = {"ATCG", "GATC", "CGAT", "AAAA", "TATA"};

    bool all_verified = true;
    for (const auto& pattern : queries) {
        auto [L, R] = backward_search(index, pattern, false);
        int fm_count = (L != -1 && R >= L) ? (R - L + 1) : 0;
        int expected_count = naive_search_count(sequence, pattern);

        std::cout << "Query: '" << pattern << "' -> FM-Index Matches: " << fm_count
                  << " | Naive Matches: " << expected_count;
        if (fm_count == expected_count) {
            std::cout << " [VERIFIED PASSED]\n";
        } else {
            std::cout << " [FAILED]\n";
            all_verified = false;
        }
    }

    std::cout << "\nVerification Status: " 
              << (all_verified ? "SUCCESS (All 5 queries match brute-force ground truth)" : "VERIFICATION ERROR") 
              << "\n";

    // 4. Parallel Benchmark Batch Demonstration
    int num_bench_queries = (argc > 2) ? std::stoi(argv[2]) : 10000;
    int read_length = (argc > 3) ? std::stoi(argv[3]) : 20;

    std::cout << "\n--- Parallel Batch Benchmark (" << num_bench_queries << " Queries, " << read_length << " bp) ---\n";
    std::vector<std::string> bench_queries(num_bench_queries);
    for (int i = 0; i < num_bench_queries; ++i) {
        std::string q = "";
        for (int j = 0; j < read_length; ++j) {
            q += sequence[(i * 7 + j * 13) % sequence.length()];
        }
        bench_queries[i] = q;
    }

    long long total_matches = 0;
    auto start_search = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_matches)
    for (int i = 0; i < num_bench_queries; ++i) {
        auto [L, R] = backward_search(index, bench_queries[i], false);
        if (L != -1 && R >= L) {
            total_matches += (R - L + 1);
        }
    }

    auto end_search = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double, std::milli>(end_search - start_search).count();
    double throughput = (num_bench_queries / (search_time / 1000.0));

    std::cout << "Total Queries Processed: " << num_bench_queries << "\n";
    std::cout << "Total Matching Occurrences: " << total_matches << "\n";
    std::cout << "Parallel Execution Time: " << search_time << " ms\n";
    std::cout << "Search Throughput: " << throughput << " queries/sec\n";
    std::cout << "========================================================\n";

    return 0;
}
