#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <numeric>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
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
std::string read_fasta(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Notice: Could not open FASTA file at " << filepath << std::endl;
        return "";
    }
    std::string sequence = "";
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '>') continue;
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
struct FMIndex {
    std::string bwt;
    std::vector<int> SA;
    std::vector<int> C;
    std::vector<std::vector<int>> Occ;
    FMIndex() : C(6, 0) {}
};
FMIndex build_fm_index(const std::string& text) {
    FMIndex index;
    int n = text.size();
    index.SA.resize(n);
    std::iota(index.SA.begin(), index.SA.end(), 0);
    std::sort(index.SA.begin(), index.SA.end(), [&text](int i, int j) {
        return text.compare(i, std::string::npos, text, j, std::string::npos) < 0;
    });
    index.bwt.resize(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (index.SA[i] == 0) {
            index.bwt[i] = '$';
        } else {
            index.bwt[i] = text[index.SA[i] - 1];
        }
    }
    std::vector<int> counts(5, 0);
    for (char c : text) {
        int idx = char_to_idx(c);
        if (idx != -1) counts[idx]++;
    }
    index.C[0] = 0;
    for (int i = 1; i <= 5; ++i) {
        index.C[i] = index.C[i - 1] + counts[i - 1];
    }
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
std::pair<int, int> backward_search(const FMIndex& index, const std::string& pattern) {
    int m = pattern.size();
    int c_idx = char_to_idx(pattern[m - 1]);
    if (c_idx == -1) return {-1, -1};
    int L = index.C[c_idx];
    int R = index.C[c_idx + 1] - 1;
    for (int i = m - 2; i >= 0 && L <= R; --i) {
        c_idx = char_to_idx(pattern[i]);
        if (c_idx == -1) return {-1, -1};
        L = index.C[c_idx] + index.Occ[c_idx][L];
        R = index.C[c_idx] + index.Occ[c_idx][R + 1] - 1;
    }
    if (L > R) return {-1, -1};
    return {L, R};
}
std::vector<std::string> generate_queries(const std::string& sequence, int num_queries, int read_length) {
    std::vector<std::string> queries(num_queries);
    std::mt19937 rng(42);
    int max_start = sequence.length() - read_length;
    std::uniform_int_distribution<int> dist(0, max_start);
    for (int i = 0; i < num_queries; ++i) {
        int start = dist(rng);
        queries[i] = sequence.substr(start, read_length);
    }
    return queries;
}
int main(int argc, char* argv[]) {
    std::cout << "========================================================" << std::endl;
    std::cout << "  OpenMP Parallel BWT & FM-Index DNA Pattern Searcher   " << std::endl;
    std::cout << "========================================================" << std::endl;
    #ifdef _OPENMP
    std::cout << "OpenMP Status: ENABLED (Max Threads: " << omp_get_max_threads() << ")" << std::endl;
    #else
    std::cout << "OpenMP Status: DISABLED (Running in sequential mode)" << std::endl;
    #endif
    std::string fasta_path = (argc > 1) ? argv[1] : "GCF_000005845.2_ASM584v2_genomic.fna";
    int num_queries = (argc > 2) ? std::stoi(argv[2]) : 10000;
    int read_length = (argc > 3) ? std::stoi(argv[3]) : 20;
    std::cout << "Loading FASTA file: " << fasta_path << "..." << std::endl;
    std::string sequence = read_fasta(fasta_path);
    if (sequence.empty()) {
        std::cout << "Generating synthetic benchmark genome (100,000 bp)..." << std::endl;
        std::string bases = "ACGT";
        std::mt19937 rng(123);
        for (int i = 0; i < 100000; ++i) sequence += bases[rng() % 4];
    }
    std::string text = sequence + "$";
    std::cout << "Genome Length: " << sequence.length() << " bp" << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();
    FMIndex index = build_fm_index(text);
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double, std::milli>(end_build - start_build).count();
    std::cout << "FM-Index Construction Time: " << build_time << " ms" << std::endl;
    std::cout << "Generating " << num_queries << " query reads (length=" << read_length << " bp)..." << std::endl;
    std::vector<std::string> queries = generate_queries(sequence, num_queries, read_length);
    std::vector<int> match_counts(num_queries, 0);
    long long total_matches = 0;
    double start_time = 0.0;
    #ifdef _OPENMP
    start_time = omp_get_wtime();
    #else
    auto chrono_start = std::chrono::high_resolution_clock::now();
    #endif
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_matches)
    for (int i = 0; i < num_queries; ++i) {
        auto [L, R] = backward_search(index, queries[i]);
        int count = (L != -1 && R >= L) ? (R - L + 1) : 0;
        match_counts[i] = count;
        total_matches += count;
    }
    double search_time = 0.0;
    #ifdef _OPENMP
    search_time = (omp_get_wtime() - start_time) * 1000.0;
    #else
    auto chrono_end = std::chrono::high_resolution_clock::now();
    search_time = std::chrono::duration<double, std::milli>(chrono_end - chrono_start).count();
    #endif
    std::cout << "\n--- Parallel Search Performance Results ---" << std::endl;
    std::cout << "Total Queries Processed: " << num_queries << std::endl;
    std::cout << "Total Matching Occurrences: " << total_matches << std::endl;
    std::cout << "Execution Time: " << search_time << " ms" << std::endl;
    std::cout << "Throughput: " << (num_queries / (search_time / 1000.0)) << " queries/sec" << std::endl;
    std::cout << "========================================================" << std::endl;
    return 0;
}
