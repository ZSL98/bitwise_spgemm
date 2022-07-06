#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000001c0,0x0000005001010002,0x0000000000000170\n"
".quad 0x0000000000000170,0x0000003400010007,0x0000000800000040,0x0000000000002013\n"
".quad 0x0000000000000000,0x00000000000003b8,0x2075632e6e69616d,0x0000000000000000\n"
".quad 0x010102464c457fa2,0x0002660001000733,0x10210001007200be,0x0701d00031000703\n"
".quad 0x00340534000ef500,0x0040000300380040,0x68732e0000010005,0x2e00626174727473\n"
".quad 0xf100086d79270008,0x0078646e68735f00,0x6f666e692e766e2e,0x612e6c6572af0009\n"
".quad 0x3000416e6f697463,0x0000328c0a00010f,0x2300010004000300,0x080202b10001004b\n"
".quad 0x08000000222f0a10,0x1300080800230010,0x1300081813000810,0x1300082813000820\n"
".quad 0x1300083811000830,0x1300400113004001,0x1300400113004001,0x1300400113004001\n"
".quad 0x1300400113004001,0x1300400213004002,0x1300400213004002,0x1300400213004002\n"
".quad 0x6f00400213004002,0x0001002c14000000,0x0100032e01e7002f,0x412e000100402200\n"
".quad 0x0b1f000108003000,0x004000812f040040,0x00010e00d4131113,0x063011000100c822\n"
".quad 0x01b5021600240200,0x0b5b01f000180036,0xf822000100700000,0x000100d82a000100\n"
".quad 0x000006570008081b,0x2a00010c02f80500,0x00c80817000800a8,0x380817000100052f\n"
".quad 0x8017000100062f00, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[58];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif
