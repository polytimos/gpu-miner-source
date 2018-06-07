/* Test kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2017 tpruvot
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author tpruvot 2017
 */
 
#if __ENDIAN_LITTLE__
  #define SPH_LITTLE_ENDIAN 1
#else
  #define SPH_BIG_ENDIAN 1
#endif
 
#define SPH_UPTR sph_u64
typedef unsigned int sph_u32;
typedef int sph_s32;
 
#ifndef __OPENCL_VERSION__
  typedef unsigned long long sph_u64;
#else
  typedef unsigned long sph_u64;
#endif
 
#define SPH_64 1
#define SPH_64_TRUE 1
 
#define SPH_C32(x) ((sph_u32)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n) SPH_ROTL32(x, (32 - (n)))
 
#define SPH_C64(x) ((sph_u64)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n) SPH_ROTL64(x, (64 - (n)))
 
 
#define SPH_ECHO_64 1
 
#ifndef SPH_LUFFA_PARALLEL
  #define SPH_LUFFA_PARALLEL 0
#endif
 
#include "skein.cl"
#include "shabal.cl"
#include "echo.cl"
#include "luffa.cl"
#include "fugue.cl"
#include "streebog.cl"
 
#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)
 
#if SPH_BIG_ENDIAN
  #define DEC64LE(x) SWAP8(*(const __global sph_u64 *) (x));
#else
  #define DEC64LE(x) (*(const __global sph_u64 *) (x));
#endif
 
#define SHL(x, n) ((x) << (n))
#define SHR(x, n) ((x) >> (n))
 
typedef union {
  unsigned char h1[64];
  uint h4[16];
  ulong h8[8];
} hash_t;
 
#define SWAP8_OUTPUT(x)  SWAP8(x)
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global unsigned char* block, __global hash_t* hashes)
{
  uint gid = get_global_id(0);
  __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
 
  // input skein 80
 
  sph_u64 M0, M1, M2, M3, M4, M5, M6, M7;
  sph_u64 M8, M9;
 
  M0 = DEC64LE(block + 0);
  M1 = DEC64LE(block + 8);
  M2 = DEC64LE(block + 16);
  M3 = DEC64LE(block + 24);
  M4 = DEC64LE(block + 32);
  M5 = DEC64LE(block + 40);
  M6 = DEC64LE(block + 48);
  M7 = DEC64LE(block + 56);
  M8 = DEC64LE(block + 64);
  M9 = DEC64LE(block + 72);
  ((uint*)&M9)[1] = SWAP4(gid);
 
  sph_u64 h0 = SPH_C64(0x4903ADFF749C51CE);
  sph_u64 h1 = SPH_C64(0x0D95DE399746DF03);
  sph_u64 h2 = SPH_C64(0x8FD1934127C79BCE);
  sph_u64 h3 = SPH_C64(0x9A255629FF352CB1);
  sph_u64 h4 = SPH_C64(0x5DB62599DF6CA7B0);
  sph_u64 h5 = SPH_C64(0xEABE394CA9D5C3F4);
  sph_u64 h6 = SPH_C64(0x991112C71A75B523);
  sph_u64 h7 = SPH_C64(0xAE18A40B660FCC33);
 
  // h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
  sph_u64 h8 = SPH_C64(0xcab2076d98173ec4);
 
  sph_u64 t0 = 64;
  sph_u64 t1 = SPH_C64(0x7000000000000000);
  sph_u64 t2 = SPH_C64(0x7000000000000040); // t0 ^ t1;
 
  sph_u64 p0 = M0;
  sph_u64 p1 = M1;
  sph_u64 p2 = M2;
  sph_u64 p3 = M3;
  sph_u64 p4 = M4;
  sph_u64 p5 = M5;
  sph_u64 p6 = M6;
  sph_u64 p7 = M7;
 
  TFBIG_4e(0);
  TFBIG_4o(1);
  TFBIG_4e(2);
  TFBIG_4o(3);
  TFBIG_4e(4);
  TFBIG_4o(5);
  TFBIG_4e(6);
  TFBIG_4o(7);
  TFBIG_4e(8);
  TFBIG_4o(9);
  TFBIG_4e(10);
  TFBIG_4o(11);
  TFBIG_4e(12);
  TFBIG_4o(13);
  TFBIG_4e(14);
  TFBIG_4o(15);
  TFBIG_4e(16);
  TFBIG_4o(17);
  TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, 18);
 
  h0 = M0 ^ p0;
  h1 = M1 ^ p1;
  h2 = M2 ^ p2;
  h3 = M3 ^ p3;
  h4 = M4 ^ p4;
  h5 = M5 ^ p5;
  h6 = M6 ^ p6;
  h7 = M7 ^ p7;
 
  // second part with nonce
  p0 = M8;
  p1 = M9;
  p2 = p3 = p4 = p5 = p6 = p7 = 0;
  t0 = 80;
  t1 = SPH_C64(0xB000000000000000);
  t2 = SPH_C64(0xB000000000000050); // t0 ^ t1;
  h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
 
  TFBIG_4e(0);
  TFBIG_4o(1);
  TFBIG_4e(2);
  TFBIG_4o(3);
  TFBIG_4e(4);
  TFBIG_4o(5);
  TFBIG_4e(6);
  TFBIG_4o(7);
  TFBIG_4e(8);
  TFBIG_4o(9);
  TFBIG_4e(10);
  TFBIG_4o(11);
  TFBIG_4e(12);
  TFBIG_4o(13);
  TFBIG_4e(14);
  TFBIG_4o(15);
  TFBIG_4e(16);
  TFBIG_4o(17);
  TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, 18);
  h0 = p0 ^ M8;
  h1 = p1 ^ M9;
  h2 = p2;
  h3 = p3;
  h4 = p4;
  h5 = p5;
  h6 = p6;
  h7 = p7;
 
  // close
  t0 = 8;
  t1 = SPH_C64(0xFF00000000000000);
  t2 = SPH_C64(0xFF00000000000008); // t0 ^ t1;
  h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
 
  p0 = p1 = p2 = p3 = p4 = p5 = p6 = p7 = 0;
 
  TFBIG_4e(0);
  TFBIG_4o(1);
  TFBIG_4e(2);
  TFBIG_4o(3);
  TFBIG_4e(4);
  TFBIG_4o(5);
  TFBIG_4e(6);
  TFBIG_4o(7);
  TFBIG_4e(8);
  TFBIG_4o(9);
  TFBIG_4e(10);
  TFBIG_4o(11);
  TFBIG_4e(12);
  TFBIG_4o(13);
  TFBIG_4e(14);
  TFBIG_4o(15);
  TFBIG_4e(16);
  TFBIG_4o(17);
  TFBIG_ADDKEY(p0, p1, p2, p3, p4, p5, p6, p7, h, t, 18);
 
  hash->h8[0] = p0;
  hash->h8[1] = p1;
  hash->h8[2] = p2;
  hash->h8[3] = p3;
  hash->h8[4] = p4;
  hash->h8[5] = p5;
  hash->h8[6] = p6;
  hash->h8[7] = p7;
 
  //if (!gid) printf("SK80 %02x..%02x\n", (uint) ((uchar*)&p0)[0], (uint) ((uchar*)&p7)[7]);

  barrier(CLK_GLOBAL_MEM_FENCE);
}
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search1(__global hash_t* hashes)
{
  uint gid = get_global_id(0);
  uint offset = get_global_offset(0);
  __global hash_t *hash = &(hashes[gid-offset]);
 
  // shabal
  sph_u32 A00 = A_init_512[0], A01 = A_init_512[1], A02 = A_init_512[2], A03 = A_init_512[3], A04 = A_init_512[4], A05 = A_init_512[5], A06 = A_init_512[6], A07 = A_init_512[7],
    A08 = A_init_512[8], A09 = A_init_512[9], A0A = A_init_512[10], A0B = A_init_512[11];
  sph_u32 B0 = B_init_512[0], B1 = B_init_512[1], B2 = B_init_512[2], B3 = B_init_512[3], B4 = B_init_512[4], B5 = B_init_512[5], B6 = B_init_512[6], B7 = B_init_512[7],
    B8 = B_init_512[8], B9 = B_init_512[9], BA = B_init_512[10], BB = B_init_512[11], BC = B_init_512[12], BD = B_init_512[13], BE = B_init_512[14], BF = B_init_512[15];
  sph_u32 C0 = C_init_512[0], C1 = C_init_512[1], C2 = C_init_512[2], C3 = C_init_512[3], C4 = C_init_512[4], C5 = C_init_512[5], C6 = C_init_512[6], C7 = C_init_512[7],
    C8 = C_init_512[8], C9 = C_init_512[9], CA = C_init_512[10], CB = C_init_512[11], CC = C_init_512[12], CD = C_init_512[13], CE = C_init_512[14], CF = C_init_512[15];
  sph_u32 M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
  sph_u32 Wlow = 1, Whigh = 0;
 
  M0 = hash->h4[0];
  M1 = hash->h4[1];
  M2 = hash->h4[2];
  M3 = hash->h4[3];
  M4 = hash->h4[4];
  M5 = hash->h4[5];
  M6 = hash->h4[6];
  M7 = hash->h4[7];
  M8 = hash->h4[8];
  M9 = hash->h4[9];
  MA = hash->h4[10];
  MB = hash->h4[11];
  MC = hash->h4[12];
  MD = hash->h4[13];
  ME = hash->h4[14];
  MF = hash->h4[15];
 
  INPUT_BLOCK_ADD;
  XOR_W;
  APPLY_P;
  INPUT_BLOCK_SUB;
  SWAP_BC;
  INCR_W;
 
  M0 = 0x80;
  M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;
 
  INPUT_BLOCK_ADD;
  XOR_W;
  APPLY_P;
 
  for (unsigned i = 0; i < 3; i ++)
  {
    SWAP_BC;
    XOR_W;
    APPLY_P;
  }
 
  hash->h4[0] = B0;
  hash->h4[1] = B1;
  hash->h4[2] = B2;
  hash->h4[3] = B3;
  hash->h4[4] = B4;
  hash->h4[5] = B5;
  hash->h4[6] = B6;
  hash->h4[7] = B7;
  hash->h4[8] = B8;
  hash->h4[9] = B9;
  hash->h4[10] = BA;
  hash->h4[11] = BB;
  hash->h4[12] = BC;
  hash->h4[13] = BD;
  hash->h4[14] = BE;
  hash->h4[15] = BF;

  //if (!gid) printf("SHABAL %02x..%02x\n", (uint) hash->h1[0], (uint) hash->h1[31]);
  //if (!gid) printf("SHABAL %02x..%02x\n", (uint) ((uchar*)&B0)[0], (uint) ((uchar*)&BF)[15]);
 
  barrier(CLK_GLOBAL_MEM_FENCE);
}
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search2(__global hash_t* hashes)
{
  uint gid = get_global_id(0);
  uint offset = get_global_offset(0);
  hash_t hash;
  __global hash_t *hashp = &(hashes[gid-offset]);
 
  __local sph_u32 AES0[256], AES1[256], AES2[256], AES3[256];
 
  int init = get_local_id(0);
  int step = get_local_size(0);
 
  for (int i = init; i < 256; i += step)
  {
    AES0[i] = AES0_C[i];
    AES1[i] = AES1_C[i];
    AES2[i] = AES2_C[i];
    AES3[i] = AES3_C[i];
  }
 
  barrier(CLK_LOCAL_MEM_FENCE);
 
  for (int i = 0; i < 8; i++)
    hash.h8[i] = hashes[gid-offset].h8[i];
 
  // echo
  sph_u64 W00, W01, W10, W11, W20, W21, W30, W31, W40, W41, W50, W51, W60, W61, W70, W71, W80, W81, W90, W91, WA0, WA1, WB0, WB1, WC0, WC1, WD0, WD1, WE0, WE1, WF0, WF1;
  sph_u64 Vb00, Vb01, Vb10, Vb11, Vb20, Vb21, Vb30, Vb31, Vb40, Vb41, Vb50, Vb51, Vb60, Vb61, Vb70, Vb71;
  Vb00 = Vb10 = Vb20 = Vb30 = Vb40 = Vb50 = Vb60 = Vb70 = 512UL;
  Vb01 = Vb11 = Vb21 = Vb31 = Vb41 = Vb51 = Vb61 = Vb71 = 0;
 
  sph_u32 K0 = 512;
  sph_u32 K1 = 0;
  sph_u32 K2 = 0;
  sph_u32 K3 = 0;
 
  W00 = Vb00;
  W01 = Vb01;
  W10 = Vb10;
  W11 = Vb11;
  W20 = Vb20;
  W21 = Vb21;
  W30 = Vb30;
  W31 = Vb31;
  W40 = Vb40;
  W41 = Vb41;
  W50 = Vb50;
  W51 = Vb51;
  W60 = Vb60;
  W61 = Vb61;
  W70 = Vb70;
  W71 = Vb71;
  W80 = hash.h8[0];
  W81 = hash.h8[1];
  W90 = hash.h8[2];
  W91 = hash.h8[3];
  WA0 = hash.h8[4];
  WA1 = hash.h8[5];
  WB0 = hash.h8[6];
  WB1 = hash.h8[7];
  WC0 = 0x80;
  WC1 = 0;
  WD0 = 0;
  WD1 = 0;
  WE0 = 0;
  WE1 = 0x200000000000000;
  WF0 = 0x200;
  WF1 = 0;
 
  for (unsigned u = 0; u < 10; u ++)
    BIG_ROUND;
 
  hashp->h8[0] = SWAP8(hash.h8[0] ^ Vb00 ^ W00 ^ W80);
  hashp->h8[1] = SWAP8(hash.h8[1] ^ Vb01 ^ W01 ^ W81);
  hashp->h8[2] = SWAP8(hash.h8[2] ^ Vb10 ^ W10 ^ W90);
  hashp->h8[3] = SWAP8(hash.h8[3] ^ Vb11 ^ W11 ^ W91);
  hashp->h8[4] = SWAP8(hash.h8[4] ^ Vb20 ^ W20 ^ WA0);
  hashp->h8[5] = SWAP8(hash.h8[5] ^ Vb21 ^ W21 ^ WA1);
  hashp->h8[6] = SWAP8(hash.h8[6] ^ Vb30 ^ W30 ^ WB0);
  hashp->h8[7] = SWAP8(hash.h8[7] ^ Vb31 ^ W31 ^ WB1);
 
  //if (!gid) printf("ECHO %02x..%02x\n", (uint) hashp->h1[0], (uint) hashp->h1[31]);

  barrier(CLK_GLOBAL_MEM_FENCE);
}
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search3(__global hash_t* hashes)
{
  uint gid = get_global_id(0);
  __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
 
  // luffa
 
  sph_u32 V00 = SPH_C32(0x6d251e69), V01 = SPH_C32(0x44b051e0), V02 = SPH_C32(0x4eaa6fb4), V03 = SPH_C32(0xdbf78465), V04 = SPH_C32(0x6e292011), V05 = SPH_C32(0x90152df4), V06 = SPH_C32(0xee058139), V07 = SPH_C32(0xdef610bb);
  sph_u32 V10 = SPH_C32(0xc3b44b95), V11 = SPH_C32(0xd9d2f256), V12 = SPH_C32(0x70eee9a0), V13 = SPH_C32(0xde099fa3), V14 = SPH_C32(0x5d9b0557), V15 = SPH_C32(0x8fc944b3), V16 = SPH_C32(0xcf1ccf0e), V17 = SPH_C32(0x746cd581);
  sph_u32 V20 = SPH_C32(0xf7efc89d), V21 = SPH_C32(0x5dba5781), V22 = SPH_C32(0x04016ce5), V23 = SPH_C32(0xad659c05), V24 = SPH_C32(0x0306194f), V25 = SPH_C32(0x666d1836), V26 = SPH_C32(0x24aa230a), V27 = SPH_C32(0x8b264ae7);
  sph_u32 V30 = SPH_C32(0x858075d5), V31 = SPH_C32(0x36d79cce), V32 = SPH_C32(0xe571f7d7), V33 = SPH_C32(0x204b1f67), V34 = SPH_C32(0x35870c6a), V35 = SPH_C32(0x57e9e923), V36 = SPH_C32(0x14bcb808), V37 = SPH_C32(0x7cde72ce);
  sph_u32 V40 = SPH_C32(0x6c68e9be), V41 = SPH_C32(0x5ec41e22), V42 = SPH_C32(0xc825b7c7), V43 = SPH_C32(0xaffb4363), V44 = SPH_C32(0xf5df3999), V45 = SPH_C32(0x0fc688f1), V46 = SPH_C32(0xb07224cc), V47 = SPH_C32(0x03e86cea);
 
  DECL_TMP8(M);
 
  M0 = (hash->h4[1]);
  M1 = (hash->h4[0]);
  M2 = (hash->h4[3]);
  M3 = (hash->h4[2]);
  M4 = (hash->h4[5]);
  M5 = (hash->h4[4]);
  M6 = (hash->h4[7]);
  M7 = (hash->h4[6]);
 
  for(uint i = 0; i < 5; i++)
  {
    MI5;
    LUFFA_P5;
 
    if(i == 0)
    {
      M0 = (hash->h4[9]);
      M1 = (hash->h4[8]);
      M2 = (hash->h4[11]);
      M3 = (hash->h4[10]);
      M4 = (hash->h4[13]);
      M5 = (hash->h4[12]);
      M6 = (hash->h4[15]);
      M7 = (hash->h4[14]);
    }
    else if(i == 1)
    {
      M0 = 0x80000000;
      M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    }
    else if(i == 2)
      M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
    else if(i == 3)
    {
      hash->h4[0] = V00 ^ V10 ^ V20 ^ V30 ^ V40;
      hash->h4[1] = V01 ^ V11 ^ V21 ^ V31 ^ V41;
      hash->h4[2] = V02 ^ V12 ^ V22 ^ V32 ^ V42;
      hash->h4[3] = V03 ^ V13 ^ V23 ^ V33 ^ V43;
      hash->h4[4] = V04 ^ V14 ^ V24 ^ V34 ^ V44;
      hash->h4[5] = V05 ^ V15 ^ V25 ^ V35 ^ V45;
      hash->h4[6] = V06 ^ V16 ^ V26 ^ V36 ^ V46;
      hash->h4[7] = V07 ^ V17 ^ V27 ^ V37 ^ V47;
    }
  }
 
  hash->h4[8] = V00 ^ V10 ^ V20 ^ V30 ^ V40;
  hash->h4[9] = V01 ^ V11 ^ V21 ^ V31 ^ V41;
  hash->h4[10] = V02 ^ V12 ^ V22 ^ V32 ^ V42;
  hash->h4[11] = V03 ^ V13 ^ V23 ^ V33 ^ V43;
  hash->h4[12] = V04 ^ V14 ^ V24 ^ V34 ^ V44;
  hash->h4[13] = V05 ^ V15 ^ V25 ^ V35 ^ V45;
  hash->h4[14] = V06 ^ V16 ^ V26 ^ V36 ^ V46;
  hash->h4[15] = V07 ^ V17 ^ V27 ^ V37 ^ V47;
 
  //if (!gid) printf("LUFFA %02x..%02x\n", (uint) hash->h1[0], (uint) hash->h1[31]);
  //if (!gid) printf("LUFFA %02x..%02x\n", (uint) ((uchar*)&M0)[0], (uint) ((uchar*)&M7)[0]);

  barrier(CLK_GLOBAL_MEM_FENCE);
}
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search4(__global hash_t* hashes)
{
  uint gid = get_global_id(0);
  uint offset = get_global_offset(0);
  __global hash_t *hash = &(hashes[gid-offset]);
 
  //mixtab
  __local sph_u32 mixtab0[256], mixtab1[256], mixtab2[256], mixtab3[256];
  int init = get_local_id(0);
  int step = get_local_size(0);
  for (int i = init; i < 256; i += step)
  {
    mixtab0[i] = mixtab0_c[i];
    mixtab1[i] = mixtab1_c[i];
    mixtab2[i] = mixtab2_c[i];
    mixtab3[i] = mixtab3_c[i];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
 
  // fugue
  sph_u32 S00, S01, S02, S03, S04, S05, S06, S07, S08, S09;
  sph_u32 S10, S11, S12, S13, S14, S15, S16, S17, S18, S19;
  sph_u32 S20, S21, S22, S23, S24, S25, S26, S27, S28, S29;
  sph_u32 S30, S31, S32, S33, S34, S35;
 
  ulong fc_bit_count = (sph_u64) 0x200;
 
  S00 = S01 = S02 = S03 = S04 = S05 = S06 = S07 = S08 = S09 = S10 = S11 = S12 = S13 = S14 = S15 = S16 = S17 = S18 = S19 = 0;
  S20 = SPH_C32(0x8807a57e); S21 = SPH_C32(0xe616af75); S22 = SPH_C32(0xc5d3e4db); S23 = SPH_C32(0xac9ab027);
  S24 = SPH_C32(0xd915f117); S25 = SPH_C32(0xb6eecc54); S26 = SPH_C32(0x06e8020b); S27 = SPH_C32(0x4a92efd1);
  S28 = SPH_C32(0xaac6e2c9); S29 = SPH_C32(0xddb21398); S30 = SPH_C32(0xcae65838); S31 = SPH_C32(0x437f203f);
  S32 = SPH_C32(0x25ea78e7); S33 = SPH_C32(0x951fddd6); S34 = SPH_C32(0xda6ed11d); S35 = SPH_C32(0xe13e3567);
 
  FUGUE512_3((hash->h4[0x0]), (hash->h4[0x1]), (hash->h4[0x2]));
  FUGUE512_3((hash->h4[0x3]), (hash->h4[0x4]), (hash->h4[0x5]));
  FUGUE512_3((hash->h4[0x6]), (hash->h4[0x7]), (hash->h4[0x8]));
  FUGUE512_3((hash->h4[0x9]), (hash->h4[0xA]), (hash->h4[0xB]));
  FUGUE512_3((hash->h4[0xC]), (hash->h4[0xD]), (hash->h4[0xE]));
  FUGUE512_3((hash->h4[0xF]), as_uint2(fc_bit_count).y, as_uint2(fc_bit_count).x);
 
  // apply round shift if necessary
  int i;
 
  for (i = 0; i < 32; i ++)
  {
    ROR3;
    CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
    SMIX(S00, S01, S02, S03);
  }
 
  for (i = 0; i < 13; i ++)
  {
    S04 ^= S00;
    S09 ^= S00;
    S18 ^= S00;
    S27 ^= S00;
    ROR9;
    SMIX(S00, S01, S02, S03);
    S04 ^= S00;
    S10 ^= S00;
    S18 ^= S00;
    S27 ^= S00;
    ROR9;
    SMIX(S00, S01, S02, S03);
    S04 ^= S00;
    S10 ^= S00;
    S19 ^= S00;
    S27 ^= S00;
    ROR9;
    SMIX(S00, S01, S02, S03);
    S04 ^= S00;
    S10 ^= S00;
    S19 ^= S00;
    S28 ^= S00;
    ROR8;
    SMIX(S00, S01, S02, S03);
  }
  S04 ^= S00;
  S09 ^= S00;
  S18 ^= S00;
  S27 ^= S00;
 
  hash->h4[0] = SWAP4(S01);
  hash->h4[1] = SWAP4(S02);
  hash->h4[2] = SWAP4(S03);
  hash->h4[3] = SWAP4(S04);
  hash->h4[4] = SWAP4(S09);
  hash->h4[5] = SWAP4(S10);
  hash->h4[6] = SWAP4(S11);
  hash->h4[7] = SWAP4(S12);
  hash->h4[8] = SWAP4(S18);
  hash->h4[9] = SWAP4(S19);
  hash->h4[10] = SWAP4(S20);
  hash->h4[11] = SWAP4(S21);
  hash->h4[12] = SWAP4(S27);
  hash->h4[13] = SWAP4(S28);
  hash->h4[14] = SWAP4(S29);
  hash->h4[15] = SWAP4(S30);

  barrier(CLK_GLOBAL_MEM_FENCE);
}
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search5(__global hash_t* hashes, __global uint* output, const ulong target)
{
  uint gid = get_global_id(0);
  __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
 
  // Streebog
 
  __local sph_u64 lT[8][256];
 
  for(int i=0; i<8; i++) {
    for(int j=0; j<256; j++) lT[i][j] = T[i][j];
  }
 
  __local unsigned char lCC[12][64];
  __local void*    vCC[12];
  __local sph_u64* sCC[12];
 
  for(int i=0; i<12; i++) {
    for(int j=0; j<64; j++) lCC[i][j] = CC[i][j];
  }
 
  for(int i=0; i<12; i++) {
    vCC[i] = lCC[i];
  }
  for(int i=0; i<12; i++) {
    sCC[i] = vCC[i];
  }
 
  sph_u64 message[8];
  message[0] = (hash->h8[0]);
  message[1] = (hash->h8[1]);
  message[2] = (hash->h8[2]);
  message[3] = (hash->h8[3]);
  message[4] = (hash->h8[4]);
  message[5] = (hash->h8[5]);
  message[6] = (hash->h8[6]);
  message[7] = (hash->h8[7]);
 
  sph_u64 out[8];
  sph_u64 len = 512;
  GOST_HASH_512(message, len, out);

  //if (!gid) printf("STREEBOG %02x..%02x\n", (uint) hash->h1[0], (uint) hash->h1[31]);
  //if (!gid) printf("STREEBOG %02x..%02x\n", (uint) ((uchar*)out)[0], (uint) ((uchar*)out)[31]);
 
  if (out[3] <= target)
    output[atomic_inc(output+0xFF)] = gid;
}
