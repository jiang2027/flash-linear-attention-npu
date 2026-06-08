/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_causal_conv1d_bwd.cpp
 * \brief Causal Conv1D backward kernel unit tests
 */

#include "causal_conv1d_bwd_tiling.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" void causal_conv1d_bwd(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR dy,
    GM_ADDR initial_state, GM_ADDR dht, GM_ADDR dx,
    GM_ADDR dw, GM_ADDR db, GM_ADDR dh0,
    GM_ADDR workspace, GM_ADDR tiling);

class CausalConv1dBwdKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./causal_conv1d_bwd_data/");
    }
    static void TearDownTestCase() {}

    static const string rootPath;
    static const string dataPath;
};

const string CausalConv1dBwdKernelTest::rootPath = "../../../../../";
const string CausalConv1dBwdKernelTest::dataPath =
    rootPath + "attention/causal_conv1d_bwd/tests/ut/op_kernel/causal_conv1d_bwd_data";

TEST_F(CausalConv1dBwdKernelTest, test_kernel_float32_no_activation)
{
    const uint32_t B = 2, T = 128, D = 256, W = 4;
    const uint32_t BT = 64, BD = 64;
    const uint32_t numBlksD = D / BD;
    const uint32_t NTPerSeq = (T + BT - 1) / BT;
    const uint32_t numChunks = NTPerSeq * B;
    const uint32_t blockNum = 40;

    system("cd ./causal_conv1d_bwd_data/ && python3 gen_data.py '(2, 128, 256, 4)' float32 0");

    size_t xByteSize = B * T * D * sizeof(float);
    size_t yByteSize = B * T * D * sizeof(float);
    size_t wByteSize = W * D * sizeof(float);
    size_t dyByteSize = B * T * D * sizeof(float);
    size_t dxByteSize = B * T * D * sizeof(float);
    size_t dwByteSize = W * D * sizeof(float);
    size_t dbByteSize = D * sizeof(float);
    size_t tilingByteSize = sizeof(CausalConv1dBwdTilingData);

    const uint32_t SYS_WS = 16 * 1024 * 1024;
    size_t wsByteSize = SYS_WS;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yByteSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(wByteSize);
    uint8_t *dy = (uint8_t *)AscendC::GmAlloc(dyByteSize);
    uint8_t *dx = (uint8_t *)AscendC::GmAlloc(dxByteSize);
    uint8_t *dw = (uint8_t *)AscendC::GmAlloc(dwByteSize);
    uint8_t *db = (uint8_t *)AscendC::GmAlloc(dbByteSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(wsByteSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingByteSize);

    string path = "./causal_conv1d_bwd_data";
    ReadFile(path + "/float32_x.bin", xByteSize, x, xByteSize);
    ReadFile(path + "/float32_weight.bin", wByteSize, w, wByteSize);
    ReadFile(path + "/float32_dy.bin", dyByteSize, dy, dyByteSize);

    memset(dw, 0, dwByteSize);
    memset(db, 0, dbByteSize);

    auto *td = reinterpret_cast<CausalConv1dBwdTilingData *>(tiling);
    td->B = B; td->T = T; td->D = D; td->W = W;
    td->activation = 0; td->hasWeight = 1; td->hasBias = 1;
    td->useInitialState = 0; td->useFinalState = 0;
    td->BT = BT; td->BD = BD;
    td->numBlksD = numBlksD; td->numChunks = numChunks;
    td->blockNum = blockNum;
    td->chunkPerCore = numChunks / blockNum;
    td->tailChunk = numChunks % blockNum;
    td->batchPerCore = 1; td->tailBatch = 0;
    td->workspaceSize = SYS_WS;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(causal_conv1d_bwd, blockNum,
        x, nullptr, w, dy, nullptr, nullptr,
        dx, dw, db, nullptr, ws, tiling);

    WriteFile(path + "/float32_dx_output.bin", dx, dxByteSize);
    WriteFile(path + "/float32_dw_output.bin", dw, dwByteSize);
    WriteFile(path + "/float32_db_output.bin", db, dbByteSize);

    AscendC::GmFree(x); AscendC::GmFree(y); AscendC::GmFree(w);
    AscendC::GmFree(dy); AscendC::GmFree(dx); AscendC::GmFree(dw);
    AscendC::GmFree(db); AscendC::GmFree(ws); AscendC::GmFree(tiling);

    system("cd ./causal_conv1d_bwd_data/ && python3 compare_data.py float32");
}

TEST_F(CausalConv1dBwdKernelTest, test_kernel_float32_with_silu)
{
    const uint32_t B = 2, T = 128, D = 256, W = 4;
    const uint32_t BT = 64, BD = 64;
    const uint32_t numBlksD = D / BD;
    const uint32_t NTPerSeq = (T + BT - 1) / BT;
    const uint32_t numChunks = NTPerSeq * B;
    const uint32_t blockNum = 40;

    system("cd ./causal_conv1d_bwd_data/ && python3 gen_data.py '(2, 128, 256, 4)' float32 1");

    size_t xByteSize = B * T * D * sizeof(float);
    size_t yByteSize = B * T * D * sizeof(float);
    size_t wByteSize = W * D * sizeof(float);
    size_t dyByteSize = B * T * D * sizeof(float);
    size_t dxByteSize = B * T * D * sizeof(float);
    size_t dwByteSize = W * D * sizeof(float);
    size_t dbByteSize = D * sizeof(float);
    size_t tilingByteSize = sizeof(CausalConv1dBwdTilingData);

    const uint32_t SYS_WS = 16 * 1024 * 1024;
    size_t wsByteSize = SYS_WS;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yByteSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(wByteSize);
    uint8_t *dy = (uint8_t *)AscendC::GmAlloc(dyByteSize);
    uint8_t *dx = (uint8_t *)AscendC::GmAlloc(dxByteSize);
    uint8_t *dw = (uint8_t *)AscendC::GmAlloc(dwByteSize);
    uint8_t *db = (uint8_t *)AscendC::GmAlloc(dbByteSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(wsByteSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingByteSize);

    string path = "./causal_conv1d_bwd_data";
    ReadFile(path + "/float32_x.bin", xByteSize, x, xByteSize);
    ReadFile(path + "/float32_y.bin", yByteSize, y, yByteSize);
    ReadFile(path + "/float32_weight.bin", wByteSize, w, wByteSize);
    ReadFile(path + "/float32_dy.bin", dyByteSize, dy, dyByteSize);

    memset(dw, 0, dwByteSize);
    memset(db, 0, dbByteSize);

    auto *td = reinterpret_cast<CausalConv1dBwdTilingData *>(tiling);
    td->B = B; td->T = T; td->D = D; td->W = W;
    td->activation = 1; td->hasWeight = 1; td->hasBias = 1;
    td->useInitialState = 0; td->useFinalState = 0;
    td->BT = BT; td->BD = BD;
    td->numBlksD = numBlksD; td->numChunks = numChunks;
    td->blockNum = blockNum;
    td->chunkPerCore = numChunks / blockNum;
    td->tailChunk = numChunks % blockNum;
    td->batchPerCore = 1; td->tailBatch = 0;
    td->workspaceSize = SYS_WS;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(causal_conv1d_bwd, blockNum,
        x, y, w, dy, nullptr, nullptr,
        dx, dw, db, nullptr, ws, tiling);

    WriteFile(path + "/float32_dx_output.bin", dx, dxByteSize);
    WriteFile(path + "/float32_dw_output.bin", dw, dwByteSize);
    WriteFile(path + "/float32_db_output.bin", db, dbByteSize);

    AscendC::GmFree(x); AscendC::GmFree(y); AscendC::GmFree(w);
    AscendC::GmFree(dy); AscendC::GmFree(dx); AscendC::GmFree(dw);
    AscendC::GmFree(db); AscendC::GmFree(ws); AscendC::GmFree(tiling);

    system("cd ./causal_conv1d_bwd_data/ && python3 compare_data.py float32");
}