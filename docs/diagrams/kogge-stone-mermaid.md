<!--
docs/diagrams/kogge-stone-mermaid.md — Mermaid diagram showing the Kogge-Stone carry propagation design.
-->

```mermaid
graph TD
    A[Balanced input trits] -->|Stage 1| B[Prefix carry computation]
    B -->|Stage 2| C[Carry generation network]
    C -->|Stage 3| D[Sum & normalization]
    D -->|Stage 4| E[Balanced output]
    subgraph Stage annotations
        B --> F[Prefix + lookahead]
        C --> G[Log-time propagation]
        D --> H[Canonical tryte output]
    end
```
