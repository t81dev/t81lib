```mermaid
flowchart LR
    subgraph Core [t81::core]
        limb[l1:limb (48 trits)]
        bigint[bigint (limb slices)]
    end
    subgraph HighLevel [Umbrella helpers]
        Int[t81::Int]
        Float[t81::Float / FloatN]
        BigInt[t81::BigInt alias]
        Ratio[t81::Ratio]
        Vector[t81::Vector]
    end
    limb --> Int
    limb --> Float
    bigint --> BigInt
    BigInt --> Ratio
    Vector --> Float
    Float --> Ratio
    Vector --> Int
    subgraph Ops [Arithmetic & GEMM]
        GEMM[t81::linalg::gemm_ternary]
        Fixed[t81::Fixed<N>]
    end
    Int --> Ops
    Float --> Ops
    Vector --> GEMM
    Fixed --> GEMM
    click Float "docs/api-overview.md" "See the helper summary"
```
