// lm_baseline.cc
// Very simple transformer‑like language model for the pd02_app assignment.
// Compiles with: clang++ -O3 -std=c++17 lm_baseline.cc -o lm_baseline
// Run:   ./lm_baseline <vocab> <emb_dim> <seq_len> <gen_len>
// Example: ./lm_baseline 32 64 128 30

#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------------
// Helper functions
// ------------------------------------------------------------------
float rand_float(mt19937& rng){
    uniform_real_distribution<float> dist(-1.f,1.f);
    return dist(rng);
}

// softmax over 1‑D vector (in‑place)
void softmax(vector<float>& v){
    float mx = *max_element(v.begin(), v.end());
    float sum = 0.f;
    vector<float> expv(v.size());
    for (size_t i=0;i<v.size();++i){
        expv[i]=expf(v[i]-mx);
        sum+=expv[i];
    }
    for (size_t i=0;i<v.size();++i) v[i]=expv[i]/sum;
}

// ------------------------------------------------------------------
// Simple feed‑forward block with weight matrices W1, W2
// ------------------------------------------------------------------
struct FFBlock{
    int emb, vocab_;
    vector<vector<float>> W1; // (emb_dim x emb_dim)
    vector<vector<float>> W2; // (emb_dim * vocab x emb_dim)
    FFBlock(int emb,int vocab): emb(emb), vocab_(vocab){
        mt19937 rng(42);
        W1.assign(emb, vector<float>(emb,0));
        W2.assign(emb, vector<float>(emb*vocab,0));
        for (auto &row:W1) for (float &x:row) x=rand_float(rng);
        for (auto &row:W2) for (float &x:row) x=rand_float(rng);
    }
    // single input token embedding (emb dim) -> hidden
    void forward(const vector<float>& x, vector<float>& h){
        h.assign(W1[0].size(),0.f);
        for (size_t k=0;k<W1.size();++k){
            float sum=0.f;
            for (size_t j=0;j<W1[0].size();++j) sum+=x[j]*W1[k][j];
            h[k]=tanh(sum);
        }
    }
    // hidden -> vocab logits
    void head(const vector<float>& h, vector<float>& logits){
        logits.assign(vocab_,0.f);
        for (size_t k=0;k<vocab_;++k){
            float sum=0.f;
            for (size_t j=0;j<W2.size();++j){
                // reuse first emb dim for vocab k (simplified)
                sum+=h[j % W2[0].size()]*W2[j][j%W2[0].size()+j];
            }
            logits[k]=sum;
        }
    }
};

// ------------------------------------------------------------------
// Main driver
// ------------------------------------------------------------------
int main(int argc,char**argv){
    if(argc!=5){
        cerr<<"Usage: "<<argv[0]<<
              " <vocab_size> <emb_dim> <seq_len> <gen_len>\n";
        return 1;
    }
    int VOCAB=atoi(argv[1]);
    int EMB=atoi(argv[2]);
    int SEQLEN=atoi(argv[3]);
    int GENLEN=atoi(argv[4]);

    mt19937 rng(12345);
    FFBlock model(EMB,VOCAB);

    // ------------------- prefill (build cache) -------------------
    // Randomly initialise decoder hidden state (all zeros here)
    vector<float> hidden(EMB,0.f);
    vector<vector<float>> token_embeds(SEQLEN);
    // Dummy embedding: random vector per token
    for(int i=0;i<SEQLEN;++i){
        token_embeds[i].resize(EMB);
        for(int j=0;j<EMB;++j) token_embeds[i][j]=rand_float(rng);
    }

    // encode each time step through model
    for(int t=0;t<SEQLEN;++t){
        model.forward(token_embeds[t], hidden);
    }

    // ------------------- decode (autoregressive) -----------------
    vector<float> input_token(VOCAB,0.f); // one‑hot placeholder
    vector<float> out_logits(VOCAB,0.f);
    mt19937 gen_rng(98765);
    uniform_real_distribution<float> uni(0.f,1.f);

    for(int step=0; step<GENLEN; ++step){
        // embed current token (simple one‑hot -> random embed for demo)
        int vocab_id = int(log(uni(gen_rng))*log(uni(gen_rng)+1)); // map to [0,VOCAB-1]
        for(int k=0;k<VOCAB;++k) input_token[k]= (k==vocab_id?1.f:0.f);
        // Embed: linear combo of embeddings (here we just reuse token_embeds matrix)
        // For brevity we generate a random embed consistent with early step
        vector<float> embed(EMB);
        if(step<SEQLEN){
            embed=token_embeds[step];
        }else{
            for(int j=0;j<EMB;++j) embed[j]=rand_float(gen_rng);
        }
        model.forward(embed, hidden);
        model.head(hidden, out_logits);
        softmax(out_logits);
        // sample (greedy) token id
        float max_p=0.f; int best=-1;
        for(int i=0;i<VOCAB;++i){
            if(out_logits[i]>max_p){ max_p=out_logits[i]; best=i; }
        }
        // update one‑hot input for next step
        fill(input_token.begin(), input_token.end(), 0.f);
        input_token[best]=1.f;
    }

    // simple verification: print final vocabulary distribution
    /* double norm=0;
    for(float p:out_logits) norm+=p*p; 
    cerr<<"Final softmax norm (should ≈1): "<<norm<<'\n';*/ 
    double sum = 0.0;
    for (float p : out_logits) sum += p;
    cerr << "Softmax sum (should ≈1): " << sum << '\n';

    float max_p = *max_element(out_logits.begin(), out_logits.end());
    cerr << "max probability: " << max_p << '\n';
    return 0;
}