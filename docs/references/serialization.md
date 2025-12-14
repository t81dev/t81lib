<!--
docs/references/serialization.md — Notes on persisting balanced-ternary data.
-->

# Serialization notes

`bigint::to_bytes()` / `bigint::from_bytes()` already define a deterministic binary layout (1 byte for the sign bit, a fixed-width counter for the limb count, and each limb’s canonical 16-byte encoding). Because the helpers work with `std::span<const std::uint8_t>`, you can wire them directly into FlatBuffers tables, MsgPack extensions, or nlohmann/json adapters without copying extra metadata. The header size and limb count are derived from `std::numeric_limits<std::uint32_t>` so buffer lengths stay predictable across platforms.

## FlatBuffers / schema-friendly blobs

If you store `t81::core::bigint` inside a FlatBuffers union/table, treat `bigint::to_bytes()` as the payload and keep the byte vector inside a `flatbuffers::Vector<uint8_t>`. On deserialize, pass the `FlatBuffers` span straight into `bigint::from_bytes()`. Because the layout already includes the sign marker and limb count, the FlatBuffers schema only needs a single `bytes:[ubyte]` field, e.g.:

```text
table BigInt {
  data:[ubyte];
}
```

and your C++ adapter simply calls `bigint::from_bytes({buffer.data(), buffer.size()})`.

## MsgPack / binary extension points

MsgPack extension types (via `msgpack::type::ext`) are ideal for the 1+N-byte header format emitted by `to_bytes()`. Register a custom converter so that serialization writes the `std::vector<uint8_t>` returned by `to_bytes()` and deserialization slices the payload back into `std::span<const uint8_t>` before calling `from_bytes()`. Because `std::hash<t81::core::bigint>` is stable, you can compare the hash of the deserialized value with the hash of the original bytes to verify integrity before reusing the value in unordered maps or caches.

## nlohmann/json and hybrid text+binary workflows

nlohmann/json supports binary blobs through the `json::binary_t` alias. Wrap the `to_bytes()` result in `nlohmann::json::binary_t` and emit it as part of a larger document (e.g., alongside metadata describing the conversion threshold or dataset). When reading back, use `json::binary_t`’s `data()` pointer to build a `std::span<const std::uint8_t>` for `from_bytes()`. This keeps the JSON human-readable while keeping the balanced-ternary digits packed into the compact base-3^48 limbs.

## Tips

1. Pass the `std::span<const std::uint8_t>` you obtain from FlatBuffers/MsgPack/JSON directly into `bigint::from_bytes()` to avoid extra copies.
2. Keep the `bigint::to_bytes()` output along with any CLI metadata you need (threshold, dtype) so your AI checkpoints can be reloaded with matching `t81-qat` or `t81 convert` (or `t81-convert`) settings.
3. After deserializing, you can call `std::hash<t81::core::bigint>` and compare it with the stored hash (or the hash computed before serialization) to guard against corruption before reusing the value in unordered containers or caches.
