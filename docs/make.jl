using Documenter, DistilBERT

makedocs(
    modules=[DistilBERT],
    sitename="DistilBERT.jl",
    checkdocs=:exports,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(
    repo="github.com/JuliaTransformers/DistilBERT.jl.git",
)
