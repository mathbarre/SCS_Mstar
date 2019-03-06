using DelimitedFiles
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


spirals = readdir("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Spiral/")
spirals = spirals[2:end]

Spiral = [readdlm("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Spiral/$(spirals[i])",Int64) for i = 1:length(spirals)]

lines = readdir("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Lines/")
lines = lines[2:end]

Line = [readdlm("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Lines/$(lines[i])",Int64) for i = 1:length(lines)]

dim = 128
m = dim*dim

nb_Spirals = 5
nb_Lines = 10
wt = wavelet(WT.haar,WT.Lifting)
function greedyForms(m,iter_sample,wt,nb_Spirals,nb_Lines)
    idx = []
    chosen_spirals = []
    for i = 1:nb_Spirals
        g = [randn(m) for i in 1:iter_sample]
        msts = Inf*ones(length(Spiral))
        for j = 1:length(Spiral)
            if !(j in chosen_spirals)
                msts[j] = mstar(m,Int64.([idx;Spiral[j][:]]),g,wt,1.4,1e-3)
            end
        end
        println(msts)
        new_spiral = argmin(msts)
        idx = [idx;Spiral[new_spiral][:]]
        chosen_spirals = [chosen_spirals;new_spiral]
        println("spirals")
        println(chosen_spirals)
    end 

    chosen_lines = []
    for i = 1:nb_Lines
        g = [randn(m) for i in 1:iter_sample]
        msts = Inf*ones(length(Line))
        for j = 1:length(Line)
            if !(j in chosen_lines)
                msts[j] = mstar(m,Int64.([idx;Line[j][:]]),g,wt,1.4,1e-3)
            end
        end
        new_line = argmin(msts)
        idx = [idx;Line[new_line][:]]
        chosen_lines = [chosen_lines;new_line]
        println("lines")
        println(chosen_lines)
    end 
    return idx   
end

println("start")
greedyForms(m,10,wt,10,10)