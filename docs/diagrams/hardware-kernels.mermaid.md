```mermaid
flowchart TB
    tryte[Packed trytes (limbs)]
    load[Load registers (AVX/NEON)]
    mask[Mask & expand trits]
    multiply[Multiply columns]
    accumulate[Accumulate into FP32/BF16]
    store[Store to output buffer]
    tryte --> load --> mask --> multiply --> accumulate --> store
    style load stroke:#333,stroke-width:1px
    style mask stroke:#f66,stroke-width:1px
    style multiply stroke:#36c,stroke-width:1px
```
