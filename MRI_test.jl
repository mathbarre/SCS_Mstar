
using Distributed 
@everywhere push!(LOAD_PATH, "/Users/mathieubarre/Documents/mstar/withImage/scs_like")
@everywhere using my_SCS
@everywhere using SCS_Operators
@everywhere using Wavelets
import LinearAlgebra.dot



function mstar(m::Int64,idx::Array{Int64,1},g::Array{Array{Float64,1},1},wt::GLS{Wavelets.WT.PerBoundary},alpha::Float64,eps::Float64)::Float64
    iter = length(g)
    @everywhere (c,myfft,usefull,scale,sigma,E) = FFTspace($m,$idx,$wt) 
    bs = [Fg(g[i][1:(2*sum(usefull))],usefull,scale,"NT",wt) for i in 1:iter]
    @time x = @distributed (+) for i in 1:iter
        SCSsolve(myfft,c,[bs[i];-bs[i]],alpha,eps,eps,eps,sigma = sigma,E=E)
    end
    return x/iter
end
function mstar_noNull(m::Int64,idx::Array{Int64,1},g::Array{Array{Float64,1},1},wt::GLS{Wavelets.WT.PerBoundary},alpha::Float64,eps::Float64)::Float64
    iter = length(g)
    @everywhere (c,myfft,usefull,scale,sigma,E) = FFTspace($m,$idx,$wt) 
   
    @time x = @distributed (+) for i in 1:iter
        SCSsolve(myfft,c,[g[i];-g[i]],alpha,eps,eps,eps,sigma = 1/sqrt(2*m),E=E)
    end
    return x/iter
end


function admissible(current::Int64,m::Int64,direction::Array{Int64,1},step::Int64)::Array{Array{Int64,1},1}
    abscisse = Int64(mod(current-1,sqrt(m))+1)
    ordonnee = Int64(floor((current-1)/sqrt(m)))
    mvt = [[1;1],[1;-1],[1;0],[0;1],[0;-1],[-1;1],[-1;-1],[-1;0]]
    candidats = [[Int64((abscisse+i*d[1])+sqrt(m)*(ordonnee+i*d[2])) for i in 1:step if prod(([abscisse;ordonnee]+i*d).>[0;-1])&&prod(([abscisse;ordonnee]+i*d).<[sqrt(m)+1;sqrt(m)])] for d in mvt if (dot(d,direction)>0)]
    return candidats 
end

function direction(m::Int64,beg::Int64,ending::Int64)::Array{Int64,1}
    abscisse_beg = Int64(mod(beg-1,sqrt(m))+1)
    ordonnee_beg = Int64(floor((beg-1)/sqrt(m)))
    abscisse_end = Int64(mod(ending-1,sqrt(m))+1)
    ordonnee_end = Int64(floor((ending-1)/sqrt(m)))
    return Int64.(sign.([abscisse_end-abscisse_beg;ordonnee_end-ordonnee_beg]))
end



function MRI_search(m::Int64,d0::Array{Int64,1},step::Int64,init::Int64,max_length::Int64,iter_sample::Int64,wt::GLS{Wavelets.WT.PerBoundary},lambda::Float64,thresh::Float64)::Tuple{Array{Int64,1},Array{Int64,1}}
    path = [init]
    unique = [init]
    candidats = admissible(init,m,d0,step)
    file = open("/Users/mathieubarre/Documents/mstar/withImage/scs_like/path64_potential_$(lambda)_$(thresh).txt","w")
    write(file,string(init),"\n")
    close(file)
    println(candidats)
    while length(unique) < max_length
        println(length(unique))
        nb_candidats = sum([length(c) for c in candidats])
        
        current = path[end]
        println(current)
        if nb_candidats == 0
            path = [path;init]
            new_d = [Int64((1-floor(2*rand()))*sign(rand()-0.5));Int64((1-floor(2*rand()))*sign(rand()-0.5))]
            candidats = admissible(path[end],m,new_d,step)
            file = open("/Users/mathieubarre/Documents/mstar/withImage/scs_like/path64_potential_$(lambda)_$(thresh).txt","a")
            write(file,string(init),"\n")
            close(file)
        elseif nb_candidats == 1
            for c in candidats
                if length(c) != 0
                    for c_ in c
                        path = [path;c_]
                        if !(c_ in unique) 
                            unique = [unique;c_]
                        end
                        file = open("/Users/mathieubarre/Documents/mstar/withImage/scs_like/path64_potential_$(lambda)_$(thresh).txt","a")
                        write(file,string(c_),"\n")
                        close(file)
                    end
                end
            end
            new_d = direction(m,current,path[end])
            candidats = admissible(path[end],m,new_d,step)
        else
            g = [randn(m) for i in 1:iter_sample]
            min_mstar = 1e6
            to_add = []
            c_to_add = []
            for c in candidats
                if length(c) != 0
                    unique_ = Array(unique)
                    for c_ in c
                        if !(c_ in unique_) 
                            unique_ = [unique_;c_]
                        end
                    end
                    abscisse = Int64(mod(c[1]-1,sqrt(m))+1)
                    ordonnee = Int64(floor((c[1]-1)/sqrt(m)))
                    current_mstar = mstar(m,unique_,g,wt,1.4,1e-3)-lambda/(1+min(sqrt((ordonnee-init)^2 + (abscisse-init)^2),thresh))
                    println(current_mstar)
                    if current_mstar < min_mstar
                        min_mstar = current_mstar
                        to_add = unique_
                        c_to_add = c
                    end

                end
            end
            unique = to_add
            for c_ in c_to_add
                path = [path;c_]
                file = open("/Users/mathieubarre/Documents/mstar/withImage/scs_like/path64_potential_$(lambda)_$(thresh).txt","a")
                write(file,string(c_),"\n")
                close(file)
            end
            new_d = direction(m,current,path[end])
            candidats = admissible(path[end],m,new_d,step)
        end 
    end
    return (path,unique)
end

m = 32*32
d0 = [1;1]
step_ = 5
iter_sample = 1
max_length = 600
init = Int64(floor(m/2)+sqrt(m)/2 +1)
wt = wavelet(WT.haar,WT.Lifting)
(path,path_) = MRI_search(m,d0,step_,init,max_length,iter_sample,wt,100*5*sqrt(2),5*sqrt(2))

using PyPlot

fig = ones(Int64(sqrt(m)),Int64(sqrt(m)))
fig[path_] .= 0
imshow(fig,cmap = "Greys" )
show()