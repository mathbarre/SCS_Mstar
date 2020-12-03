module my_SCS

import IterativeSolvers.minres
import IterativeSolvers.cg
using LinearMaps
import LinearAlgebra.dot,LinearAlgebra.norm

export SCSsolve

function SCSsolve(Afun,c::Array{Float64,1},b::Array{Float64,1},alph::Float64,tol_p::Float64,tol_d::Float64,tol_g::Float64;sigma::Float64=1.0,rho::Float64=1.0,D::Array{Float64,1}=ones(length(b)),E::Array{Float64,1}=ones(length(c)),verbose::Bool=false)::Array{Float64,1}
    n = size(c,1)
    m = size(b,1)
    u = [zeros(n+m);1]
    v = [zeros(n+m);1]
    c = rho*E.*c
    b = sigma*D.*b
    h=[c;b]
    h_ = [c;-b]
    Afun_scale(x,s)= begin
        if s == "NT"
            return D.*Afun(E.*x,"NT") 
        else
            return E.*Afun(D.*x,"T")
        end
    end
    Mfun_(x) = [x[1:n]+Afun_scale(x[(n+1):(n+m)],"T");Afun_scale(x[1:n],"NT")-x[(n+1):(n+m)]]
    Mfun = LinearMap(Mfun_,n+m;issymmetric=true)
    #use minres with h_ because minres needs a symmetric matrix as argument
    Minv_h = minres(Mfun,h_;tol=1e-8)
    #println(norm(Mfun_(Minv_h) -h_))
    
    p = 1e10
    d = 1e10
    g = 1e10
    iter = 1
    while norm(p) > tol_p*(1+norm(b)) || norm(d) > tol_d*(1+norm(c)) ||  abs(g) > tol_g*(1+abs(dot(c,u[1:n])/u[n+m+1])+abs(dot(b,u[(n+1):(n+m)])/u[n+m+1]))
        u_ = SubspaceProj(u+v,Afun_scale,c,b,Minv_h)
        #u__ = [eye(n) A' c;-A eye(m) b;-c' -b' 1]\(u+v);
        u = alph*u_ + (1-alph)*u - v
        #projection on C
        u[(n+1):end] = max.(0,u[(n+1):end])
        v = v - alph*u_ + alph*u

        
        normalize = 1
        if u[n+m+1] != 0
            normalize = u[n+m+1]
        end
        p = (Afun_scale(u[1:n]/normalize,"NT") +v[(n+1):(n+m)]/normalize - b)/sigma ./D
        d = (Afun_scale(u[(n+1):(n+m)]/normalize,"T") +c)/rho ./E
        g = (dot(c,u[1:n])/normalize + dot(b,u[(n+1):(n+m)])/normalize)/(rho*sigma)
        
        if verbose
            if mod(iter,50) == 0
                println("")
                println(norm(p)/(1+norm(b)))
                println(norm(d)/(1+norm(c)))
                println(abs(g)/(1+abs(dot(c,u[1:n])/u[n+m+1])+abs(dot(b,u[(n+1):(n+m)])/u[n+m+1])))
                println(sum(u.^2)/size(u,1))
            end
        end
        iter += 1
        
    end
    println(u[n+m+1])
    x =u[1:n]/u[n+m+1]
    x = E.*x/sigma
    #res = dot(c,x)
    return x

end

function SubspaceProj(w::Array{Float64,1},Afun,c::Array{Float64,1},b::Array{Float64,1},Minv_h::Array{Float64,1})::Array{Float64,1}
    #               [ I   A' c]
    #       (I+Q) = [-A   I  b]
    #               [-c' -b' 1]
    # 
    #       solves  (I+Q)*u_ = w
    h=[c;b]
    n = size(c,1)
    m = size(b,1)
    wx = w[1:n]
    wy = w[(n+1):(n+m)]
    wt = w[n+m+1]
    
    AAfun_(x) = x+Afun(Afun(x,"NT"),"T");
    AAfun = LinearMap(AAfun_,n;issymmetric=true,isposdef=true)
    #real update
    zx = cg(AAfun,wx - Afun(wy,"T");tol=1e-12)
    #here assume A'A = I
    #zx = (wx - Afun(wy,"T"))/2
    zy = wy + Afun(zx,"NT");
    
    u_ = zeros(m+n+1)
    u_[1:(n+m)] = [zx;zy] - (wt + (dot(h,[zx;zy]) - wt*dot(h,Minv_h))/(1 +dot(h,Minv_h)))*Minv_h 
    u_[n+m+1] = wt + dot(c,u_[1:n]) + dot(b,u_[(n+1):(n+m)])
    return u_
end

end