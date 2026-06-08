/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling.h
 * \brief Tiling data structure for kernel UT CPU simulation
 */

#ifndef CAUSAL_CONV1D_BWD_TILING_UT_H_
#define CAUSAL_CONV1D_BWD_TILING_UT_H_

#include <cstring>
#include "../../../op_kernel/causal_conv1d_bwd_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#ifdef __NPU_TILING__
inline __aicore__ void InitTilingData(const __gm__ uint8_t *tiling, CausalConv1dBwdTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (size_t i = 0; i < sizeof(CausalConv1dBwdTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(const uint8_t *tiling, CausalConv1dBwdTilingData *constData)
{
    std::memcpy(constData, tiling, sizeof(CausalConv1dBwdTilingData));
}
#endif // __NPU_TILING__

#define GET_TILING_DATA(tilingData, tilingArg) \
    CausalConv1dBwdTilingData tilingData;      \
    InitTilingData(reinterpret_cast<const uint8_t *>(tilingArg), &tilingData)

#endif  // CAUSAL_CONV1D_BWD_TILING_UT_H_