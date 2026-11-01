// Serial portfolio search. C++17 standard library only.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
struct Data { std::vector<std::string> names, dates; Matrix prices; };
struct Stats { double annual_return=0, volatility=0, sharpe=0, total=0; };
struct Candidate { std::vector<int> units; Stats train; std::size_t id; };

std::string trim(std::string s) {
    auto a=s.find_first_not_of(" \t\r\n");
    if(a==std::string::npos) return "";
    return s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out; std::stringstream ss(line); std::string s;
    while(std::getline(ss,s,',')) out.push_back(trim(s));
    if(!line.empty() && line.back()==',') out.push_back("");
    return out;
}
double number(const std::string& s) {
    std::size_t end; double x=std::stod(s,&end);
    if(end!=s.size() || !std::isfinite(x)) throw std::runtime_error("Invalid number: "+s);
    return x;
}
Data read_csv(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open "+path);
    std::string line; if(!std::getline(in,line)) throw std::runtime_error("Empty CSV");
    auto h=split(line);
    if(h.size()<2 || h[0]!="Date") throw std::runtime_error("Header must be Date,STOCK1,STOCK2,...");
    Data d; d.names.assign(h.begin()+1,h.end());
    for(std::size_t i=0;i<d.names.size();++i) {
        if(d.names[i].empty()) throw std::runtime_error("Empty stock name");
        for(std::size_t j=0;j<i;++j) if(d.names[i]==d.names[j]) throw std::runtime_error("Duplicate stock name");
    }
    std::size_t row=1;
    while(std::getline(in,line)) {
        ++row; if(trim(line).empty()) continue;
        auto cells=split(line);
        if(cells.size()!=h.size()) throw std::runtime_error("Wrong column count at row "+std::to_string(row));
        if(cells[0].empty() || (!d.dates.empty() && cells[0]<=d.dates.back()))
            throw std::runtime_error("Dates must be unique, increasing ISO YYYY-MM-DD strings");
        std::vector<double> p;
        for(std::size_t i=1;i<cells.size();++i) {
            double x=number(cells[i]); if(x<=0) throw std::runtime_error("Prices must be positive"); p.push_back(x);
        }
        d.dates.push_back(cells[0]); d.prices.push_back(std::move(p));
    }
    if(d.prices.size()<21) throw std::runtime_error("Need at least 21 price rows");
    return d;
}
Data demo() {
    Data d; d.names={"SYN_A","SYN_B","SYN_C","SYN_D","SYN_E","SYN_F"};
    std::mt19937_64 rng(77123); std::normal_distribution<double> normal(0,1);
    std::vector<double> p(d.names.size(),100);
    for(int t=0;t<757;++t) {
        if(t) {
            double market=normal(rng);
            for(std::size_t i=0;i<p.size();++i) {
                double vol=0.007+0.002*i, drift=0.00015+0.00004*i;
                p[i]*=std::exp(drift-0.5*vol*vol+vol*(0.65*market+std::sqrt(1-0.65*0.65)*normal(rng)));
            }
        }
        d.dates.push_back("synthetic_day_"+std::to_string(t)); d.prices.push_back(p);
    }
    return d;
}
Stats evaluate(const Matrix& returns, std::size_t begin, std::size_t end,
               const std::vector<int>& units, double rf) {
    std::vector<std::size_t> active;
    for(std::size_t i=0;i<units.size();++i) if(units[i]) active.push_back(i);
    double mean=0, m2=0, log_growth=0; std::size_t count=0;
    for(std::size_t t=begin;t<end;++t) {
        double r=0;
        for(auto i:active) r+=(units[i]/1000.0)*returns[t][i];
        ++count; double delta=r-mean; mean+=delta/count; m2+=delta*(r-mean);
        log_growth+=std::log1p(r);
    }
    Stats s; s.annual_return=252*mean;
    s.volatility=std::sqrt(std::max(0.0,m2/(count-1))*252);
    // Treat effectively constant portfolios as unrankable by Sharpe.
    s.sharpe=s.volatility>1e-12 ? (s.annual_return-rf)/s.volatility : -1e100;
    s.total=std::expm1(log_growth); return s;
}
bool better(const Candidate& a,const Candidate& b) {
    if(a.train.sharpe!=b.train.sharpe) return a.train.sharpe>b.train.sharpe;
    return a.id<b.id;
}
std::size_t integer(const std::string& s) {
    if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("Expected nonnegative integer: "+s);
    auto x=std::stoull(s); if(x>std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Integer too large");
    return static_cast<std::size_t>(x);
}
int main(int argc,char** argv) {
    try {
        std::string input,output="portfolios.csv";
        std::size_t samples=20000,top=10,max_holdings=100,seed=12345;
        double fraction=0.7,rf=0;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            if(a=="--help") {
                std::cout<<"Usage: ./portfolio [--input prices.csv | --demo] [--samples 20000]\n"
                         <<"  [--top 10] [--max-holdings 100] [--seed 12345]\n"
                         <<"  [--train-fraction 0.7] [--risk-free 0] [--output portfolios.csv]\n"
                         <<"No input: deterministic synthetic demo. Annual risk-free rate is a decimal.\n";
                return 0;
            }
            if(a=="--demo") { input.clear(); continue; }
            if(i+1==argc) throw std::runtime_error("Missing value for "+a);
            std::string v=argv[++i];
            if(a=="--input") input=v;
            else if(a=="--output") output=v;
            else if(a=="--samples") samples=integer(v);
            else if(a=="--top") top=integer(v);
            else if(a=="--max-holdings") max_holdings=integer(v);
            else if(a=="--seed") seed=integer(v);
            else if(a=="--train-fraction") fraction=number(v);
            else if(a=="--risk-free") rf=number(v);
            else throw std::runtime_error("Unknown option: "+a);
        }
        if(!samples || !top || !max_holdings || fraction<=0 || fraction>=1)
            throw std::runtime_error("Positive samples/top/max-holdings and 0 < train-fraction < 1 required");
        if(!input.empty() && input==output) throw std::runtime_error("Input and output paths must differ");
        Data d=input.empty()?demo():read_csv(input);
        const auto n=d.names.size(), days=d.prices.size()-1;
        const auto cut=static_cast<std::size_t>(days*fraction);
        if(cut<2 || days-cut<2) throw std::runtime_error("Each split needs at least two daily returns");
        max_holdings=std::min({max_holdings,n,std::size_t(1000)});
        top=std::min(top,samples);
        Matrix returns(days,std::vector<double>(n));
        for(std::size_t t=0;t<days;++t) for(std::size_t i=0;i<n;++i) {
            returns[t][i]=d.prices[t+1][i]/d.prices[t][i]-1;
            if(!std::isfinite(returns[t][i]) || returns[t][i]<=-1) throw std::runtime_error("Unrepresentable price return");
        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::size_t> choose_k(1,max_holdings);
        std::vector<std::size_t> indices(n); std::iota(indices.begin(),indices.end(),0);
        std::vector<Candidate> best;
        auto start=std::chrono::steady_clock::now();
        for(std::size_t id=0;id<samples;++id) {
            auto k=choose_k(rng); std::shuffle(indices.begin(),indices.end(),rng);
            // Uniform positive compositions of 1000 units, using k-1 random cuts.
            std::vector<int> cuts(999); std::iota(cuts.begin(),cuts.end(),1);
            for(std::size_t j=0;j+1<k;++j) {
                std::uniform_int_distribution<std::size_t> pick(j,998);
                std::swap(cuts[j],cuts[pick(rng)]);
            }
            cuts.resize(k-1); std::sort(cuts.begin(),cuts.end()); cuts.push_back(1000);
            Candidate c; c.units.assign(n,0); c.id=id; int previous=0;
            for(std::size_t j=0;j<k;++j) { c.units[indices[j]]=cuts[j]-previous; previous=cuts[j]; }
            c.train=evaluate(returns,0,cut,c.units,rf);
            if(best.size()==top && !better(c,best.back())) continue;
            bool duplicate=false;
            for(const auto& b:best) if(c.units==b.units) { duplicate=true; break; }
            if(duplicate) continue;
            best.insert(std::lower_bound(best.begin(),best.end(),c,better),std::move(c));
            if(best.size()>top) best.pop_back();
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream out(output); if(!out) throw std::runtime_error("Cannot write "+output);
        out<<"rank,split,annual_mean_return,annual_volatility,sharpe,total_return";
        for(const auto& name:d.names) { // Quote stock names in output CSV.
            out<<",\""; for(char ch:name) { if(ch=='\"') out<<'\"'; out<<ch; } out<<'\"';
        }
        out<<'\n'<<std::setprecision(12);
        std::cout<<(input.empty()?"SYNTHETIC DEMO (not market data)":"Historical CSV")
                 <<" | stocks="<<n<<" train_days="<<cut<<" test_days="<<days-cut
                 <<" candidates="<<samples<<" search_seconds="<<seconds<<'\n';
        std::cout<<"Training: "<<d.dates[0]<<" -> "<<d.dates[cut]
                 <<"; test: "<<d.dates[cut]<<" -> "<<d.dates.back()<<'\n';
        for(std::size_t rank=0;rank<best.size();++rank) {
            auto test=evaluate(returns,cut,days,best[rank].units,rf);
            for(int split_id=0;split_id<2;++split_id) {
                const auto& s=split_id?test:best[rank].train;
                out<<rank+1<<','<<(split_id?"test":"train")<<','<<s.annual_return<<','<<s.volatility<<',';
                if(s.sharpe<=-1e99) out<<"NA"; else out<<s.sharpe;
                out<<','<<s.total;
                for(int u:best[rank].units) out<<','<<u/1000.0;
                out<<'\n';
            }
            std::cout<<"#"<<rank+1<<" train Sharpe=";
            if(best[rank].train.sharpe<=-1e99) std::cout<<"NA"; else std::cout<<best[rank].train.sharpe;
            std::cout<<" test total return="<<std::fixed<<std::setprecision(2)<<test.total*100<<"% |";
            for(std::size_t i=0;i<n;++i) if(best[rank].units[i])
                std::cout<<' '<<d.names[i]<<'='<<std::setprecision(1)<<best[rank].units[i]/10.0<<'%';
            std::cout<<std::defaultfloat<<std::setprecision(6)<<'\n';
        }
        out.close(); if(!out) throw std::runtime_error("Failed writing output");
        std::cout<<"Saved "<<best.size()<<" distinct portfolios to "<<output<<'\n';
    } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n'; return 1; }
}
