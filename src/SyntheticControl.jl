module SyntheticControl

export SDID, NNM, RSC, elastic_net_sc, LatentRegression

using Transducers
using LinearAlgebra
using Functional
using StatsBase

include("utils.jl")
include("elastic-net.jl")
include("nnm.jl")
include("sdid.jl")
include("rsc.jl")
include("latent_regression.jl")

end # module
