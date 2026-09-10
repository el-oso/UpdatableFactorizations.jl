using Documenter, DocumenterVitepress, UpdatableFactorizations

makedocs(;
    sitename = "UpdatableFactorizations.jl",
    modules = [UpdatableFactorizations],
    repo = Documenter.Remotes.GitHub("el-oso", "UpdatableFactorizations.jl"),
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/UpdatableFactorizations.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Construction" => "construction.md",
        "Updating and downdating" => "updating.md",
        "Q representations" => "q_representations.md",
        "Benchmarks" => "benchmarks.md",
        "Provenance" => "provenance.md",
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/UpdatableFactorizations.jl",
    push_preview = true,
)
