module DistilBERTTDExt

# Package extension: wires TransformersDownloads.jl into DistilBERT.jl.
# Loaded automatically when both packages are present in the session.

using DistilBERT
using TransformersDownloads

DistilBERT._td_load_config(source::String) = TransformersDownloads.load_config(source)

DistilBERT._td_load_weights(source::String) = TransformersDownloads.load_weights(source)

DistilBERT._td_vocab(source::String) = TransformersDownloads.hf_vocab(source)

end # module DistilBERTTDExt
