module LowRankModels

# define losses, regularizers, convergence history
include("loss_and_reg.jl")
include("convergence.jl")

## to use many processes to fit a model in shared memory, use shareglrm instead of glrm
if nprocs()>1
     include("shareglrm.jl")
else
	include("glrm.jl")
end
# fancy fun on top of low rank models
include("initialize.jl")
include("cross_validate.jl")
include("fit_dataframe.jl")
#include("plot.jl")
include("fitsym.jl")
include("sgd.jl")

end # module
