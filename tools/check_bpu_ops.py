#!/usr/bin/env python3
"""S600 端侧算子体检：把 ONNX 的算子清单对上地瓜官方支持表。

数据来源（抓取日期 2026-11-04，OE 手册 v3.7.0 口径）：
  toolchain.d-robotics.cc/guide/appendix/supported_op_list/operator_support/onnx_operator_support_j6p.html
  toolchain.d-robotics.cc/guide/appendix/supported_op_list/operator_constraints_bpu/onnx_operator_constraints_list_j6p.html
j6p 即 RDK S600。表里两列：BPU 能不能跑、退回 CPU 能不能跑（N/N 表示工具链完全不支持，会挡编译）。

用法：
    ~/hb-env/bin/python tools/check_bpu_ops.py models/policy_v1_20261103.onnx [more.onnx ...]

退出码：0 全部算子可上 BPU / 1 有算子只能走 CPU / 2 有不支持或表外算子。
"""

import sys
from collections import Counter
from pathlib import Path

import onnx

# 官方 S600 ONNX 算子支持列表，Op:BPU+CPU（Y/N）
SUPPORT = (
    "Abs:YY Acos:YY Acosh:YY Add:YY AffineGrid:NN And:YY ArgMax:YY ArgMin:YY Asin:YY Asinh:YY Atan:YY "
    "Atanh:YY AveragePool:YY BatchNormalization:YY Bernoulli:NN BitShift:NY BitwiseAnd:YY "
    "BitwiseNot:YY BitwiseOr:YY BitwiseXor:YY BlackmanWindow:NN Cast:YY CastLike:NN Ceil:YY Celu:YY "
    "CenterCropPad:NN Clip:YY Col2Im:NN Compress:NN Concat:YY ConcatFromSequence:NN Constant:YY "
    "ConstantOfShape:YY Conv:YY ConvInteger:NN ConvTranspose:YY Cos:YY Cosh:YY CumSum:YY "
    "DeformConv:YY DFT:NN DepthToSpace:YY DequantizeLinear:NY Det:NN Div:YY Dropout:YY "
    "DynamicQuantizeLinear:NN Einsum:NN Elu:YY Equal:YY Erf:YY Exp:YY Expand:YY EyeLike:YY Flatten:YY "
    "Floor:YY GRU:YY Gather:YY GatherElements:YY GatherND:YY Gelu:YY Gemm:YY GlobalAveragePool:YY "
    "GlobalLpPool:NN GlobalMaxPool:YY Greater:YY GreaterOrEqual:YY GridSample:YY "
    "GroupNormalization:YY HammingWindow:NN HannWindow:NN HardSigmoid:YY HardSwish:YY Hardmax:NN "
    "Identity:YY If:NN ImageDecoder:NN InstanceNormalization:YY IsInf:NN IsNaN:NN LRN:NN LSTM:YY "
    "LayerNormalization:YY LeakyRelu:YY Less:YY LessOrEqual:YY Log:YY LogSoftmax:YY Loop:NN "
    "LpNormalization:NN LpPool:NN MatMul:YY MatMulInteger:NN Max:YY MaxPool:YY MaxRoiPool:NN "
    "MaxUnpool:NN Mean:NN MeanVarianceNormalization:NN MelWeightMatrix:NN Min:YY Mish:YY Mod:YY "
    "Mul:YY Multinomial:NN Neg:YY NegativeLogLikelihoodLoss:NN NonMaxSuppression:NY NonZero:NY Not:YY "
    "OneHot:NN Optional:NN OptionalGetElement:NN OptionalHasElement:NN Or:YY PRelu:YY Pad:YY Pow:NY "
    "QLinearConv:NN QLinearMatMul:NN QuantizeLinear:NY RNN:NN RandomNormal:NN RandomNormalLike:NN "
    "RandomUniform:NN RandomUniformLike:NN Range:NN Reciprocal:YY ReduceL1:NN ReduceL2:YY "
    "ReduceLogSum:NY ReduceLogSumExp:NN ReduceMax:YY ReduceMean:YY ReduceMin:YY ReduceProd:NY "
    "ReduceSum:YY ReduceSumSquare:NN RegexFullMatch:NN Relu:YY Reshape:YY Resize:YY "
    "ReverseSequence:NN RoiAlign:NN Round:YY STFT:NN Scan:NN Scatter_(deprecated):NN "
    "ScatterElements:YY ScatterND:YY Selu:YY SequenceAt:NN SequenceConstruct:NN SequenceEmpty:NN "
    "SequenceErase:NN SequenceInsert:NN SequenceLength:NN SequenceMap:NN Shape:YY Shrink:NN "
    "Sigmoid:YY Sign:YY Sin:YY Sinh:YY Size:YY Slice:YY Softmax:YY SoftmaxCrossEntropyLoss:NN "
    "Softplus:YY Softsign:NN SpaceToDepth:YY Split:YY SplitToSequence:NN Sqrt:YY Squeeze:YY "
    "StringConcat:NN StringNormalizer:NN StringSplit:NN Sub:YY Sum:YY Tan:YY Tanh:YY "
    "TfIdfVectorizer:NN ThresholdedRelu:YY Tile:YY TopK:YY Transpose:YY Trilu:YY Unique:NN "
    "Unsqueeze:YY Upsample_(deprecated):NN Where:YY Xor:YY"
)

# 我们模型实际用到、且官方有附加约束的算子（原文摘录，避免臆造）
CAVEATS = {
    "Gemm": "BPU 只收 int8/int16 输入（输出可为 float32）；K ≤ 8192、N ≤ 1048576",
    "LayerNormalization": "BPU 只收 int8/int16/float16，float32 不在支持类型内",
    "ReduceL2": "BPU 只收 int8/int16/float16，float32 不在支持类型内",
    "ScatterND": "float32 输入要求 scatterReduceMode=add；updates 只支持 float16",
    "Slice": "仅当 starts/ends/axes/steps 为常量才上 BPU；由 Shape 推算的动态 Slice 不在此列",
    "Expand": "类型不限，形状需可静态推导",
    "Cast": "不支持 float16/float32 → bool8",
    "ArgMin": "input 为 int32/float32 时只支持单轴",
    "OneHot": "BPU/CPU 两列均为 N：工具链完全不支持，会挡编译",
}


def table():
    parsed = {}
    for item in SUPPORT.split():
        name, flags = item.rsplit(":", 1)
        parsed[name.replace("_", "")] = (flags[0], flags[1])
    return parsed


def lookup(parsed, op):
    if op in parsed:
        return parsed[op]
    return parsed.get(op + "(deprecated)")


def check(path):
    graph = onnx.load(str(path)).graph
    counts = Counter(node.op_type for node in graph.node)
    rows, verdicts = [], set()
    for op, num in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        flags = lookup(table(), op)
        if flags is None:
            verdict, mark = "表外算子（需人工核对）", "UNKNOWN"
        elif flags == ("Y", "Y"):
            verdict, mark = "BPU + CPU", "BPU"
        elif flags == ("Y", "N"):
            verdict, mark = "仅 BPU", "BPU"
        elif flags == ("N", "Y"):
            verdict, mark = "仅 CPU（不上 BPU）", "CPU"
        else:
            verdict, mark = "工具链不支持", "UNSUPPORTED"
        verdicts.add(mark)
        rows.append((op, num, verdict, mark))

    worst = 2 if verdicts & {"UNSUPPORTED", "UNKNOWN"} else (1 if "CPU" in verdicts else 0)
    print(f"\n=== {path} ===")
    print(f"{'算子':<22}{'节点数':>6}  判定")
    for op, num, verdict, mark in rows:
        extra = f"   # {CAVEATS[op]}" if op in CAVEATS else ""
        print(f"{op:<22}{num:>6}  [{mark}] {verdict}{extra}")
    if worst == 0:
        print("结论：全部算子都在 S600 BPU 支持范围内（量化后仍受各算子类型约束，见上方注释）")
    elif worst == 1:
        print("结论：有算子只能退回 CPU 执行；整图仍可编译，但性能与前后处理开销需实测")
    else:
        print("结论：存在工具链不支持的算子，hb_compile 会拒绝该模型，需先改图")
    return worst


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    codes = [check(p) for p in argv[1:]]
    return max(codes)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
