# Architecture overview

`ARCHITECTURE.md` captures how the balanced-ternary pieces are organized so new contributors
can quickly orient themselves before reading the full specs in `doc/`.

## Pillars

- **Canonical scalar (`limb`)**  
  A 48-trit value with stable 16-byte encoding, explicit overflow/underflow handling, trit/tryte helpers,
  and deterministic hashing. Everything else builds on this base.
- **Arbitrary-precision integers (`bigint`)**  
  Sign + limb vector representation. Normalization keeps a canonical magnitude, while
  add/sub/mul/div leverage magnitude helpers plus Karatsuba when it helps.
- **Montgomery & helpers**  
  `MontgomeryContext` plus helper factories/guards allow efficient modular multiply/pow with strict
  width checks and explicit conversion paths.
- **I/O and utilities**  
  `t81::io` keeps serialization/formatting consistent, while `t81::util` supplies randomness,
  invariant checks, and debug-friendly dumps.

## Module relationships

```
     ┌───────────────┐
     │ application   │
     └──────┬────────┘
            │ includes
     ┌──────▼──────────┐
     │ t81/t81lib.hpp  │  < umbrella that re-exports
     └──────┬──────────┘
            │
   ┌────────▼────────────┐
   │ t81::core::limb     │  (unit of trits, canonical arithmetic)
   └───────┬─────────────┘
           │ uses
   ┌───────▼─────────────┐
   │ t81::core::bigint    │ (sign + limbs + magnitude helpers)
   └───────┬─────────────┘
           │ supports
   ┌───────▼─────────────┐
   │ Montgomery helpers  │ (contexts, guards, modular multiply/pow)
   └───────┬─────────────┘
           │ enabled by
   ┌───────▼──────────────┐
   │ t81::io / util        │ (formatting, parsing, randomness, invariants)
   └──────────────────────┘
```

## Engineering notes

- **Normalization invariants**: `limbs_` never stores trailing zero limbs, which ensures equality checks
  and conversions are deterministic and simplifies modular arithmetic.
- **Sign handling**: `bigint` keeps a `negative_` flag; zero resets the flag to keep the representation
  unique.
- **Wide arithmetic helpers**: Magnitude helpers (`add_magnitude`, `multiply_magnitude`, etc.) use
  `detail::limb_int128` to avoid overflow before normalizing back to canonical limbs.
  
## Further reading

For the formal semantics, refer to `doc/t81lib-spec-v1.0.0.md`. Internal design decisions are spelled out
in `doc/design/limb-design.md`, `doc/design/bigint-design.md`, and `doc/design/montgomery-design.md`.

This guide is intentionally light and developer-facing—if you need a runnable overview, `docs/index.md`
acts as the higher-level docs portal introduced in the README.
