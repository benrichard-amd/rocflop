# rocflop
ROCm FLOPs microbench

## Building
Run `make`

## Running

```
--device  ID          Use device with the given numerical ID
--devices IDS | ALL   Comma-separated list of device Ids (e.g., 1,2,3)
                      ALL for all devices
--runs    RUNS        Number of times each kernel is dispatched
--fp16                Run FP16 (VALU) test
--fp32                Run FP32 (VALU) test
--fp64                Run FP64 (VALU) test
--matfp16             Run FP16 (MFMA) test
--matfp32             Run FP32 (MFMA) test
--smatfp16            Run FP16 (SMFMAC) test
```
