export fitsgdsym, fitsgdsym!

function fitsgdsym!(glrm::GLRM; params::Params=Params(),ch::ConvergenceHistory=ConvergenceHistory("glrm"),verbose=true)
	
	### initialization
	A = glrm.A
	m,n = size(A)
	losses = glrm.losses
	rx = glrm.rx
	ry = glrm.ry
	X = glrm.X; Y = glrm.Y
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
    g = zeros(k)

    # cache views
    ve = StridedView{Float64,2,0,Array{Float64,2}}[view(X,e,:) for e=1:m]
    vf = ContiguousView{Float64,1,Array{Float64,2}}[view(Y,:,f) for f=1:n]

    minibatchsize = 10

    for iter=1:params.max_iter
        # update random row of X
        XY = X*Y
        e = sample(1:n)
        # a gradient of L wrt e
        scale!(g, 0)
        for j=1:minibatchsize
            f = sample(glrm.observed_features[e])
        	axpy!(grad(losses[f],XY[e,f],A[e,f]), vf[f], g)
        end
        # take a proximal gradient step
        ## gradient step: g = X[e,:] - alpha/l*g
        l = minibatchsize
        scale!(g, -alpha/l)
        axpy!(1,g,ve[e])
        ## prox step: X[e,:] = prox(g)
        prox!(rx,ve[e],alpha/l)
        # update corresponding column of Y
        Y[:,e] = X[e,:]'
        # record the current model, but not too often, and decrease step size
        if iter%100 == 0
            t = time() - t
            obj = objective(glrm,X,Y)
            update!(ch, t, obj)
            alpha = params.stepsize / (iter/100)
            t = time()
            if verbose
                println("Iteration $iter: objective value = $obj") 
            end
        end
    end
    # t = time() - t
    #t = time()
    #update!(ch, t, ch.objective[end])

    return glrm.X,glrm.Y,ch
end

function fitsgdsym(glrm::GLRM, args...; kwargs...)
    X0 = Array(Float64, size(glrm.X))
    Y0 = Array(Float64, size(glrm.Y))
    copy!(X0, glrm.X); copy!(Y0, glrm.Y)
    X,Y,ch = fitsgdsym!(glrm, args...; kwargs...)
    copy!(glrm.X, X0); copy!(glrm.Y, Y0)
    return X,Y,ch
end