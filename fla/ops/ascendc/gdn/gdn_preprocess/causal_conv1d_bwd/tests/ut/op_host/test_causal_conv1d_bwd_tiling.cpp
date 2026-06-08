/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_causal_conv1d_bwd_tiling.cpp
 * \brief Test for causal_conv1d_bwd tiling
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/causal_conv1d_bwd_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

class CausalConv1dBwdTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dBwdTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dBwdTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(CausalConv1dBwdTiling, test_tiling_float32_basic)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    
    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(CausalConv1dBwdTiling, test_tiling_float16_basic)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    
    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(CausalConv1dBwdTiling, test_tiling_with_silu_activation)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    
    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(CausalConv1dBwdTiling, test_tiling_large_batch)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    
    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{8, 256, 512}, {8, 256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 256, 512}, {8, 256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 256, 512}, {8, 256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 256, 512}, {8, 256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(CausalConv1dBwdTiling, test_tiling_small_dim)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    
    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{1, 16, 64}, {1, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 64}, {1, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 64}, {3, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 64}, {1, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 16, 64}, {1, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 64}, {3, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(CausalConv1dBwdTiling, test_tiling_dim_multiple_of_16)
{
    optiling::CausalConv1dBwdCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;

    gert::TilingContextPara tilingContextPara("CausalConv1dBwd",
        {
            {{{1, 16, 80}, {1, 16, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 80}, {1, 16, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 80}, {3, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 80}, {1, 16, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 16, 80}, {1, 16, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 80}, {3, 80}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{80}, {80}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"activation", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo);

    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
