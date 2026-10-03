# Third-Party Notices

This repository includes third-party source snapshots and derived model code. Each component retains its original license and applicable copyright notices. This document summarizes the bundled components; it does not replace their license texts or file-level notices.

## Bundled Dependency Sources

| Component | Declared version | Recorded snapshot revision | License scope | Archive |
|---|---|---|---|---|
| [ms-swift](https://github.com/modelscope/ms-swift) | `4.2.0.dev0` | `e97d200d1eb27f2fdfd6ba03bdd3c12dfb9533c4` | Apache-2.0 | [`swift.tar.gz`](third_party/swift.tar.gz) |
| [Transformers](https://github.com/huggingface/transformers) | `5.7.0` | `90dd8674248257e1115e531f2250fd85a32c3863` | Apache-2.0 | [`transformers.tar.gz`](third_party/transformers.tar.gz) |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | `0.5.0` | `bbc1f57e892a6e5af586c71c7733411208c75b82` | MIT (evaluation pipeline); Apache-2.0 (multimodal models and tasks) | [`lmms_eval.tar.gz`](third_party/lmms_eval.tar.gz) |

The versions above are reported by the bundled sources. These are adapted snapshots and may differ from packages with the same version on a package index. Snapshot revision identifiers describe the recorded source provenance; the [dependency manifest](third_party/manifest.json) provides the archive and file hashes for the exact contents shipped here. For lmms-eval, the revision was recorded in the source README.

Each archive contains its original `LICENSE` and a `NOTICE.release` describing the recorded adaptations. Environment setup extracts them under the corresponding `vendor/` directory, where those files remain available.

## Adaptations

- **ms-swift:** model-training integration, local data loading and packing, dependency compatibility, and credential masking. Cloud-storage-specific training integration is omitted.
- **Transformers:** distribution changes to test-token defaults and local-kernel documentation examples.
- **lmms-eval:** a selected evaluation engine and OpenAI-compatible adapters, configurable dataset locations, request/response handling, and optional media dependencies. Task definitions are supplied by this repository's `eval/` directory.

The manifest records adapted file names. Existing copyright and license notices are preserved in those files.

## Derived Model Code

The VLM implementation includes code derived from Kimi K3 and other upstream model implementations. The Kimi K3 license is retained in [`models/vlm/LICENSE`](models/vlm/LICENSE); its [upstream license source](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/LICENSE) and the source-file notices provide the applicable terms and attribution.

## Scope

Additional dependencies installed by the environment recipes retain their own licenses and package notices. Model weights and datasets are distributed separately and retain their respective terms. Original project contributions are licensed under the [Apache License 2.0](LICENSE), with no additional restrictions imposed by this project. Third-party material and derived model code retain their applicable upstream licenses.

## Public Comparison Bases and Export Patches

The following publicly resolvable commits are comparison bases for reproducing the selected source exports. They are separate from the recorded snapshot revisions above; exact ancestry is not asserted. File-level patches describe added and replaced files, while unchanged files come from the public base. Replacement content is taken from the existing source archive to avoid distributing duplicate source trees.

- **swift**: [v4.1.3](https://github.com/modelscope/ms-swift/commit/c6875ef6a962e83f01138bb239b5fb4e5e55b37f) (`c6875ef6a962e83f01138bb239b5fb4e5e55b37f`); [export patch](third_party/patches/swift.json).
- **transformers**: [v5.7.0](https://github.com/huggingface/transformers/commit/6ffbb07f93d9e44457450d1150136309b0dc966b) (`6ffbb07f93d9e44457450d1150136309b0dc966b`); [export patch](third_party/patches/transformers.json).
- **lmms_eval**: [v0.5](https://github.com/EvolvingLMMs-Lab/lmms-eval/commit/8f142bc3082100dbb39aa9b15916c586f3237d09) (`8f142bc3082100dbb39aa9b15916c586f3237d09`); [export patch](third_party/patches/lmms_eval.json).

See [dependency sources](third_party/README.md) for rebuilding an export and checking all file hashes.

## Attribution and anonymous review

Third-party attribution for anonymous review

Names of upstream authors, companies and institutions, contact addresses,
repository namespaces, public model/dataset identifiers and example URLs in
third-party source and license notices identify their original sources. They
are retained for attribution and interoperability, not as a declaration of
this submission's authorship or affiliation. Such attribution alone does not
disclose submission authors. Original licenses and copyright notices remain
unchanged. Local adaptations are recorded separately; an unverified local
remark is not assigned to upstream merely because it is in a dependency.

## Model names and public examples

GroundingPI is the public model name and serialized model_type. Its released
Python modules also use GroundingPI names. The retained Kimi K3 license and
source notices identify the vision code's provenance.

The `ilankelman.org` image URL in model documentation is a public example
resource, not an author contact detail or submission-hosted demonstration
service. The Mountchicken/Rex-Omni-Eval default paths are inherited from the
public Rex-Omni evaluation script.
