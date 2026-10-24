# Release composition

## v0.2.0

Adds the text-only `Fev4B.decide` interface, a resident reference HTTP server for `POST /v1/systemone` and `GET /v1/models`, strict typed-request/response validation, zstd decoding, TypeSafe SDK contract checks, and public GitHub documentation. Choice, Noul and Score use the existing native decision head. No checkpoint, shared adapter, tokenizer, native source, temperature, precision, routing primitive or routing cutoff changed. The routing response version is now 0.2.0.

The package verifier accepts the v0.1.0 and v0.2.0 manifest formats. All files in the selected model package are still checked by size and SHA256. Version tags remain separate; v0.1.0 is retained unchanged.

## v0.1.0

Packages the complete decision checkpoint with a shared LoRA adapter, customer-agnostic routing composition, local loader, CLI, integrity checks, synthetic examples and release documentation. Checkpoint weights, tokenizer, readout, native source, license and notices are retained without modification.

The seven routing primitives are copied byte-for-byte from the frozen runtime used for the 303-case development measurement. Full ancestry and file hashes are in `provenance.json` and `manifest.json`.
