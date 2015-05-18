export fitsym, fitsym!

function fitsym!(glrm::GLRM; params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"),verbose=true)
	
	### initialization
	A = glrm.A
	m,n = size(A)
	losses = glrm.losses
	rx = glrm.rx
	ry = glrm.ry
	# at any time, glrm.X and glrm.Y will be the best model yet found, while
	# X and Y will be the working variables
	X = copy(glrm.X); Y = copy(glrm.Y)
	k = glrm.k

    # check that we didn't initialize to zero (otherwise we will never move)
    if norm(Y) == 0 
    	Y = .1*randn(k,n) 
    end

    # step size (will be scaled below to ensure it never exceeds 1/\|g\|_2 or so for any subproblem)
    alpha = params.stepsize
    # stopping criterion: stop when decrease in objective < tol
    tol = params.convergence_tol * mapreduce(length,+,glrm.observed_features)

    # alternating updates of X and Y
    if verbose println("Fitting GLRM") end
    update!(ch, 0, objective(glrm))
    t = time()
    steps_in_a_row = 0
    g = zeros(k)

    # cache views
    ve = StridedView{Float64,2,0,Array{Float64,2}}[view(X,e,:) for e=1:m]
    vf = ContiguousView{Float64,1,Array{Float64,2}}[view(Y,:,f) for f=1:n]

    for i=1:params.max_iter
        # X update
        XY = X*Y
        for e=1:m
            # a gradient of L wrt e
            scale!(g, 0)
            for f in glrm.observed_features[e]
            	axpy!(grad(losses[f],XY[e,f],A[e,f]), vf[f], g)
            end
            # take a proximal gradient step
            ## gradient step: g = X[e,:] - alpha/l*g
            l = length(glrm.observed_features[e]) + 1
            scale!(g, -alpha/l)
            axpy!(1,g,ve[e])
            ## prox step: X[e,:] = prox(g)
            prox!(rx,ve[e],alpha/l)
        end
        # Y update
        Y[:] = X'
        obj = objective(glrm,X,Y)
        # record the best X and Y yet found
        if obj < ch.objective[end]
            t = time() - t
            update!(ch, t, obj)
            copy!(glrm.X, X); copy!(glrm.Y, Y)
            alpha = alpha * 1.05
            steps_in_a_row = max(1, steps_in_a_row+1)
            t = time()
        else
            # if the objective went up, reduce the step size, and undo the step
            alpha = alpha / max(1.5, -steps_in_a_row)
            if verbose println("obj went up to $obj; reducing step size to $alpha") end
            copy!(X, glrm.X); copy!(Y, glrm.Y)
            steps_in_a_row = min(0, steps_in_a_row-1)
        end
        # check stopping criterion
        if i>10 && (steps_in_a_row > 3 && ch.objective[end-1] - obj < tol) || alpha <= params.min_stepsize
            break
        end
        if verbose && i%10==0 
            println("Iteration $i: objective value = $(ch.objective[end])") 
        end
    end
    t = time() - t
    update!(ch, t, ch.objective[end])

    return glrm.X,glrm.Y,ch
end

function fitsym(glrm::GLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fitsym!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X,Y,ch
end