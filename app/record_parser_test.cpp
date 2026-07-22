// Standalone validation of the consumeDataStream record parser: feed it
// synthetic (payload+32-trailer)*N streams, chunked arbitrarily, and
// confirm it extracts exactly the N payloads in order. Mirrors the loop
// in platform.cpp::consumeDataStream verbatim.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <deque>
#include <cstring>
using namespace std;

static deque<uint8_t> q;   // stands in for api_recv_buf (byte-level here)
static void push_bytes(const uint8_t* p, int n){ for(int i=0;i<n;i++) q.push_back(p[i]); }

// the parser, extracted verbatim (byte-granular push to model the queue).
static void parse_stream(const vector<vector<uint8_t>>& chunks, int PAY){
  const int REC = PAY + 32;
  int rec_off = 0;
  for(const auto& ch : chunks){
    const char* p = (const char*)ch.data();
    int left = (int)ch.size();
    while(left > 0){
      if(rec_off < PAY){
        int take = PAY - rec_off; if(take>left) take=left;
        push_bytes((const uint8_t*)p, take);   // payload → queue
        p+=take; left-=take; rec_off+=take;
      } else {
        int take = REC - rec_off; if(take>left) take=left;
        p+=take; left-=take; rec_off+=take;     // trailer → drop
        if(rec_off==REC) rec_off=0;
      }
    }
  }
}

int main(){
  int fails=0;
  for(int PAY : {8192, 2048}){
    for(int CHUNKSZ : {32768, 8224, 100, 4, PAY+32, 1}){
      int N=5;
      // build N programs: payload byte i of prog k = (k*37+i)&0xff, +32 zero trailer
      vector<uint8_t> stream;
      for(int k=0;k<N;k++){
        for(int i=0;i<PAY;i++) stream.push_back((uint8_t)((k*37+i)&0xff));
        for(int i=0;i<32;i++) stream.push_back(0xEE);   // nonzero trailer to catch leaks
      }
      // slice into CHUNKSZ pieces
      vector<vector<uint8_t>> chunks;
      for(size_t o=0;o<stream.size();o+=CHUNKSZ)
        chunks.emplace_back(stream.begin()+o, stream.begin()+min(stream.size(),o+CHUNKSZ));
      q.clear();
      parse_stream(chunks, PAY);
      // expect exactly N*PAY payload bytes, matching each program
      bool ok = (int)q.size()==N*PAY;
      if(ok){
        int idx=0;
        for(int k=0;k<N&&ok;k++) for(int i=0;i<PAY&&ok;i++)
          if(q[idx++]!=(uint8_t)((k*37+i)&0xff)) ok=false;
      }
      printf("  PAY=%-5d CHUNK=%-6d -> %s (got %zu, want %d)\n",
             PAY,CHUNKSZ, ok?"PASS":"FAIL", q.size(), N*PAY);
      if(!ok) fails++;
    }
  }
  printf("record-parser: %s (%d fails)\n", fails?"FAIL":"ALL_PASS", fails);
  return fails?1:0;
}
