// Streaming SHA-256 (FIPS 180-4) for gufo-quant.
//
// `Sha256Hex` is the exact one-shot helper gufo_quant_main.cpp used to stamp
// imatrix provenance (same bytes in -> same hex out). `Sha256` adds the
// streaming form (`update`/`hex`) so --manifest can hash 100 GB inputs in one
// sequential pass with constant memory, and `Sha256File` hashes a whole file
// that way (+ its byte count, so callers never stat+read twice).
//
// v0.7: the block compression picks SHA-NI (sha256rnds2/msg1/msg2) at runtime
// on x86-64 CPUs that have it -- every Zen core and Ice Lake+ Intel. A single
// SHA-256 stream is inherently serial, so the 27B manifest's 55 GB input hash
// ran at scalar speed (~200 MB/s per thread, 251 s on the slow shard); SHA-NI
// lifts one thread past 1.5 GB/s, which moves the hash streams fully inside
// the encode's wall. Digests are bit-identical on both paths (the KATs in
// gufo_quant_stream_test and the NI-vs-scalar sweep in the lane receipts prove
// it); set GQT_SHA256_NO_NI=1 to force the portable path.

#ifndef GUFO_QUANT_SHA256_HPP_
#define GUFO_QUANT_SHA256_HPP_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define GQT_SHA_NI_X86_ 1
#include <immintrin.h>
#endif

namespace gufo::quant_tool {

// The round-constant table (FIPS 180-4 section 4.2.2), shared by both block
// implementations: the scalar loop reads it element-wise, the SHA-NI loop
// loads four entries per round group straight from it. Values verified
// 2026-10-10 against the definition (first 32 fraction bits of the cube
// roots of the first 64 primes, exact integer arithmetic), and pinned by the
// FIPS known-answer vectors in gufo_quant_stream_test Part C.
inline constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

class Sha256 {
 public:
  Sha256() { Reset(); }

  void Reset() {
    static const std::uint32_t kH[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                        0xa54ff53a, 0x510e527f, 0x9b05688c,
                                        0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, kH, sizeof(h_));
    std::memset(buf_, 0, sizeof(buf_));
    buf_len_ = 0;
    total_len_ = 0;
  }

  void Update(const void* data, std::uint64_t len) {
    const std::uint8_t* p = (const std::uint8_t*)data;
    total_len_ += len;
    while (len > 0) {
      std::uint64_t take = 64 - buf_len_;
      if (take > len) take = len;
      std::memcpy(buf_ + buf_len_, p, (std::size_t)take);
      buf_len_ += take;
      p += take;
      len -= take;
      if (buf_len_ == 64) {
        Compress(h_, buf_);
        buf_len_ = 0;
      }
    }
  }

  void Update(const std::vector<std::uint8_t>& v) {
    if (!v.empty()) Update(v.data(), v.size());
  }

  // Finalize (may be called once) and return the lowercase hex digest.
  std::string Hex() {
    std::uint64_t bitlen = total_len_ * 8;
    std::uint8_t one = 0x80;
    Update(&one, 1);
    std::uint8_t zero = 0;
    while (buf_len_ != 56) Update(&zero, 1);
    std::uint8_t len_be[8];
    for (int i = 7; i >= 0; --i) len_be[7 - i] = (std::uint8_t)(bitlen >> (8 * i));
    // Feed the length directly: buf has exactly 8 free bytes, one Block call.
    std::memcpy(buf_ + 56, len_be, 8);
    Compress(h_, buf_);
    buf_len_ = 0;
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
    return std::string(out, 64);
  }

 private:
  static inline std::uint32_t Rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  // The portable reference block (unchanged since v0.1; K now read from the
  // shared kSha256K table).
  static void BlockScalar(std::uint32_t* h, const std::uint8_t* m) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((std::uint32_t)m[4 * i] << 24) |
             ((std::uint32_t)m[4 * i + 1] << 16) |
             ((std::uint32_t)m[4 * i + 2] << 8) | ((std::uint32_t)m[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4],
                  f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      std::uint32_t ch = (e & f) ^ ((~e) & g);
      std::uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
      std::uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

#ifdef GQT_SHA_NI_X86_
  // SHA extensions block. The schedule runs in a rotating 4-register scheme
  // derived from the W recurrence (W[t] = W[t-16] + s0(W[t-15]) + W[t-7] +
  // s1(W[t-2])): msg1 gives the first two terms, alignr the W[t-7] term, msg2
  // the s1 term; the words land in w[16..63] exactly as BlockScalar computes
  // them. Compression is 16 canonical rnds2 pairs over pre-added W+K groups.
  __attribute__((target("ssse3,sse4.1,sha")))
  static void BlockShaNi(std::uint32_t* h, const std::uint8_t* m) {
    const __m128i bswap =
        _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    alignas(16) std::uint32_t w[64];
    __m128i mm[4];
    for (int j = 0; j < 4; ++j) {
      mm[j] = _mm_shuffle_epi8(
          _mm_loadu_si128((const __m128i*)(m + 16 * j)), bswap);
      _mm_store_si128((__m128i*)&w[4 * j], mm[j]);
    }
    for (int k = 0; k < 12; ++k) {
      const int s = k & 3;
      __m128i x = _mm_sha256msg1_epu32(mm[s], mm[(s + 1) & 3]);
      const __m128i t = _mm_alignr_epi8(mm[(s + 3) & 3], mm[(s + 2) & 3], 4);
      x = _mm_add_epi32(x, t);
      mm[s] = _mm_sha256msg2_epu32(x, mm[(s + 3) & 3]);
      _mm_store_si128((__m128i*)&w[16 + 4 * k], mm[s]);
    }
    __m128i st0, st1, tmp;
    tmp = _mm_loadu_si128((const __m128i*)&h[0]);
    st1 = _mm_loadu_si128((const __m128i*)&h[4]);
    tmp = _mm_shuffle_epi32(tmp, 0xB1);    /* CDAB */
    st1 = _mm_shuffle_epi32(st1, 0x1B);    /* EFGH */
    st0 = _mm_alignr_epi8(tmp, st1, 8);    /* ABEF */
    st1 = _mm_blend_epi16(st1, tmp, 0xF0); /* CDGH */
    const __m128i save0 = st0, save1 = st1;
    for (int grp = 0; grp < 16; ++grp) {
      __m128i msg =
          _mm_add_epi32(_mm_load_si128((const __m128i*)&w[4 * grp]),
                        _mm_loadu_si128((const __m128i*)&kSha256K[4 * grp]));
      st1 = _mm_sha256rnds2_epu32(st1, st0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0E);
      st0 = _mm_sha256rnds2_epu32(st0, st1, msg);
    }
    st0 = _mm_add_epi32(st0, save0);
    st1 = _mm_add_epi32(st1, save1);
    tmp = _mm_shuffle_epi32(st0, 0x1B);
    st1 = _mm_shuffle_epi32(st1, 0xB1);
    st0 = _mm_blend_epi16(tmp, st1, 0xF0);
    st1 = _mm_alignr_epi8(st1, tmp, 8);
    _mm_storeu_si128((__m128i*)&h[0], st0);
    _mm_storeu_si128((__m128i*)&h[4], st1);
  }
#endif

  // One block function, picked once per process (thread-safe magic static).
  using BlockFn = void (*)(std::uint32_t*, const std::uint8_t*);
  static BlockFn PickBlock() {
#if defined(GQT_SHA_NI_X86_)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sha") && !std::getenv("GQT_SHA256_NO_NI"))
      return &Sha256::BlockShaNi;
#endif
    return &Sha256::BlockScalar;
  }
  static void Compress(std::uint32_t* h, const std::uint8_t* m) {
    static const BlockFn fn = PickBlock();
    fn(h, m);
  }

  std::uint32_t h_[8];
  std::uint8_t buf_[64];
  std::uint64_t buf_len_ = 0;
  std::uint64_t total_len_ = 0;  // message bytes fed (excludes padding)
};

inline std::string Sha256Hex(const std::vector<std::uint8_t>& data) {
  Sha256 s;
  s.Update(data);
  return s.Hex();
}

inline std::string Sha256Str(const std::string& s) {
  Sha256 h;
  if (!s.empty()) h.Update(s.data(), s.size());
  return h.Hex();
}

// Hash a file in ONE sequential streaming pass (64 MiB chunks, constant
// memory). Sets *bytes_out to the file size counted during the same pass.
inline std::string Sha256File(const std::string& path, std::uint64_t* bytes_out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Sha256 h;
  std::vector<char> chunk(64 << 20);
  std::uint64_t n = 0;
  for (;;) {
    f.read(chunk.data(), (std::streamsize)chunk.size());
    std::streamsize got = f.gcount();
    if (got > 0) {
      h.Update(chunk.data(), (std::uint64_t)got);
      n += (std::uint64_t)got;
    }
    if (got < (std::streamsize)chunk.size()) break;
  }
  if (f.bad()) throw std::runtime_error("read error on " + path);
  if (bytes_out) *bytes_out = n;
  return h.Hex();
}

}  // namespace gufo::quant_tool

#endif  // GUFO_QUANT_SHA256_HPP_
