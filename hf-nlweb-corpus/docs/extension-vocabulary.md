# Extension vocabulary

Version `1.0`. Every published record carries
`corpus:vocabularyVersion` so a consumer can tell which revision it was written
against.

Schema.org has no `MachineLearningModel` type, so a model repository is published
as a `SoftwareApplication` and ML-specific facts are carried as
`PropertyValue` nodes under `additionalProperty`, `identifier`, or a Claim's
`additionalProperty`. The `huggingface:`, `ml:`, and `corpus:` prefixes are
**compact extension identifiers**, not a claim that schema.org defines these
properties. They are stable within a major vocabulary version.

## `huggingface:` — facts the Hub API reports

| propertyID | Where | Value | Notes |
|---|---|---|---|
| `huggingface:repo_id` | `identifier` | string | `owner/name` |
| `huggingface:sha` | `identifier` | string | Revision every derived fact was computed at |
| `huggingface:pipelineTag` | `additionalProperty` | string | Hub pipeline task |
| `huggingface:library` | `additionalProperty` | string | e.g. `transformers`, `diffusers` |
| `huggingface:downloads` | `additionalProperty` | integer | Last 30 days |
| `huggingface:downloadsAllTime` | `additionalProperty` | integer | Absent on older repositories |
| `huggingface:likes` | `additionalProperty` | integer | |
| `huggingface:trendingScore` | `additionalProperty` | number | Snapshot-time value |
| `huggingface:gatedStatus` | `additionalProperty` | `open` or `gated:<mode>` | |

## `ml:` — model facts

| propertyID | Value | Notes |
|---|---|---|
| `ml:parameterCount` | integer, `unitText: parameters` | Carries `valueReference` with `corpus:source` and `corpus:certainty` |
| `ml:weightDataTypes` | list of strings | From safetensors metadata |
| `ml:mixtureOfExperts` | boolean | Active parameters are not derivable from Hub metadata alone |
| `ml:expertCount` | integer | |
| `ml:architecture` | string | From the model config |
| `ml:modelType` | string | From the model config |
| `ml:contextLength` | integer, `unitText: tokens` | First of `max_position_embeddings`, `n_positions`, `max_sequence_length` |
| `ml:modelFormat` | list of strings | `safetensors`, `gguf`, `onnx`, … |

## `corpus:` — pipeline-derived qualification

| propertyID | Value | Notes |
|---|---|---|
| `corpus:licenseStatus` | `recognized_identifier` \| `declared_link` \| `declared_text` \| `unknown` | `unknown` is never a claim of permissive terms |
| `corpus:declaredLicenseIdentifier` | string | What the publisher declared, verbatim |
| `corpus:cardStatus` | `ok` \| `missing_readme` \| `gated` \| `too_large` \| … | |
| `corpus:conflict` | list of field names | Both source values are kept in the internal record |
| `corpus:familyId` | string | Lineage-rooted family key |
| `corpus:familyRelation` | `canonical` \| `derived_from` \| `exact_duplicate_of` \| `format_or_quantization_variant` \| `mirror_of` | |
| `corpus:discoveryScope` | `primary` \| `variant` | `variant` records stay out of broad topical results |
| `corpus:canonicalRepository` | URL | Only set when the canonical repository is in this corpus |
| `corpus:deploymentVariants` | list of URLs | Quantizations and conversions attached to the family |
| `corpus:intendedUses`, `corpus:limitations`, `corpus:deploymentNotes` | list of strings | Grounded in the card |
| `corpus:vocabularyVersion` | string | This document's version |
| `corpus:source`, `corpus:certainty` | string | Inside a `valueReference`, qualifying a measured vs inferred value |

## Claim properties

A subject specialty appears twice: as a `DefinedTerm` in `about` (what the item
is about) and as a `Claim` in `subjectOf` (why we believe it). Claim
`additionalProperty` nodes:

| propertyID | Value |
|---|---|
| `corpus:confidence` | `high` \| `medium` \| `low` |
| `corpus:evidenceType` | `trained_on` \| `fine_tuned_on` \| `evaluated_on` \| `intended_for` \| `architecture` |
| `corpus:evidenceQuote` | Verbatim span from the cleaned card at the recorded SHA |
| `corpus:sourceSection` | Heading path the quote came from |
| `corpus:taxonomyStatus` | `mapped` (in the taxonomy) \| `proposed` (free-text label, not yet vocabulary) |

`Claim.appearance.url` is the card at the pinned revision with a text fragment
anchoring the quote.

## Changing this vocabulary

Adding a property is a minor version bump. Changing the meaning or value space of
an existing property is a major bump, and requires re-emitting the corpus so no
record mixes vocabularies.
