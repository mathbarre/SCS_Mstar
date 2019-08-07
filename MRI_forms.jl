using DelimitedFiles
using Distributed 
@everywhere push!(LOAD_PATH, "/Users/mathieubarre/Documents/mstar/withImage/scs_like")
@everywhere using my_SCS
@everywhere import SCS_Operators
@everywhere using Wavelets
import LinearAlgebra.dot



function mstar(m::Int64,idx::Array{Int64,1},g::Array{Array{Float64,1},1},wt::GLS{Wavelets.WT.PerBoundary},alpha::Float64,eps::Float64)::Float64
    iter = length(g)
    @everywhere (c,myfft,usefull,scale,sigma,E) = SCS_Operators.FFTspace($m,$idx,$wt) 
    bs = [SCS_Operators.Fg(g[i][1:(2*sum(usefull))],usefull,scale,"NT",wt) for i in 1:iter]
    @time x = @distributed (+) for i in 1:iter
        SCSsolve(myfft,c,[bs[i];-bs[i]],alpha,eps,eps,eps,sigma = sigma,E=E)[1]
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
function greedyForms(m::Int64,iter_sample::Int64,wt::GLS{Wavelets.WT.PerBoundary},nb_Spirals::Int64,nb_Lines::Int64)
    idx = []
    chosen_spirals = []
    file = open("/Users/mathieubarre/Documents/These/Mstar/juliaProg/choosen_forms128.txt","a")
    write(file,"Spirals","\n")
    close(file)
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
        file = open("/Users/mathieubarre/Documents/These/Mstar/juliaProg/choosen_forms128.txt","a")
        write(file,String(chosen_spirals),"\n")
        close(file)
    end 

    chosen_lines = []
    file = open("/Users/mathieubarre/Documents/These/Mstar/juliaProg/choosen_forms128.txt","a")
    write(file,"Lines","\n")
    close(file)
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
        file = open("/Users/mathieubarre/Documents/These/Mstar/juliaProg/choosen_forms128.txt","a")
        write(file,String(chosen_lines),"\n")
        close(file)
    end 
    return idx   
end

file = open("/Users/mathieubarre/Documents/These/Mstar/juliaProg/choosen_forms128.txt","w")
write(file,"Start","\n")
close(file)
greedyForms(m,10,wt,10,10)



using PyPlot
using StatsBase
using FFTW
using ImageQualityIndexes
n=128

brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/brain128.png")
brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/brain_easy128.png")[:,:,1]
#brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/pap128.png")[:,:,1]
#brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/mriscan128.png")[:,:,1]
#brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/child128.png")[:,:,1]
brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/bambou128.png")[:,:,1]
brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/wrist_128.png")[:,:,1]
brain = imread("/Users/mathieubarre/Documents/These/Mstar/juliaProg/brain_tot_128.png")[:,:,1]
function coordToIdx(i,j,n)
    return (i-1)*n+j
end

function sphere(center,n,r)
    res = []
    for i = 1:n
        for j = 1:n
            if (i-center)^2 + (j-center)^2 <= r^2
                res= [res;coordToIdx(i,j,n)]
            end
        end
    end
    return res
end
center = n/2 +1
idx =  floor.(center .+ 50*randn(n*50,2))
#wavelet
#idx = [Int64(n*(floor(n/2))+(floor(n/2)+1));sample(1:(n*n),n*70,replace=false)]
#idx_spirals = [3;24;15;27;21;9;6;1;13;12]
idx_spirals = [3;15;9;24;21;6;27;12;13;1;7;22;25;10;4]#16;10]
idx_spirals = unique(vcat([Spiral[i][:] for i in idx_spirals][:]...))

idx_lines = []#[8;2;33;24;19;27]#;1;10;17;28]#;29;18;35;21;34;6;13;7]
idx_lines = unique(vcat([Line[i][:] for i in idx_lines][:]...))
#set_vlines = 1:3:128 
#idx_vlines = [24;13;32;19;26;22;43;11;38;41;20;14;28;23]
idx_vlines = unique(vcat([ coordToIdx(j,i,n) for i = 1:n for j = set_vlines[idx_vlines]]))
idx_tot = unique([Int64.(sphere(center,128,10));idx_spirals;idx_lines])
#idx_tot = unique([Int64.(sphere(center,128,10));idx_vlines])
hline = Int64.([coordToIdx(i,center+j,n) for i = 1:n for j =-1:3])
vline = Int64.([coordToIdx(center+j,i,n) for i = 1:n for j =-1:3])
idx = Int64.(unique([Int64(coordToIdx(center,center,n));vline;hline;idx_tot]))
function ReconstructBrain(brain,idx)
    n = size(brain,1)
    m = n*n
    (idx_r,idx_i)=SCS_Operators.cleanIndex(idx,n*n)
    wt = wavelet(WT.haar,WT.Lifting)
    #measures = SCS_Operators.myFFTBP(dwt(Float64.(brain),wt)[:],"NT",idx_r,idx_i,128*128,wt)
    d1 = fftshift(fft(ifftshift(brain)))/sqrt(m)
    #d1 = (fft((reshape(x[1:(Int(n/2))],Int(sqrt(m)),Int(sqrt(m))))))
    d1 =d1[:]
    #d1 = d1[idx_r]
    d1_r = real(d1[idx_r]) 
    d1_i = imag(d1[idx_i])
    measures = [d1_r;d1_i]*sqrt(2)
    bs = [measures;-measures;zeros(2*m)]
    (c,myfft,usefull,scale,sigma,D) = SCS_Operators.BasisPursuitOperator(m,idx,wt)
    tol = 1e-3
    alpha = 1.0
    x = SCSsolve(myfft,c,bs,alpha,tol,tol,tol,rho= 1/sqrt(m),sigma = sigma,E=ones(2*m),D=D)
    brain_rcst = idwt(reshape(x[1:m],n,n),wt)
    return brain_rcst
end
brain_rcst = ReconstructBrain(brain,idx)
figure()
imshow(brain_rcst,cmap="gray")
println(sum((brain-brain_rcst).^2)/sum(brain.^2))


rand_spirals = sample([1:15;19:27],16,replace=false)
rand_spirals = unique(vcat([Spiral[i][:] for i in rand_spirals][:]...))

rand_lines = sample([1:25 ;27:39],3,replace=false)
rand_lines = unique(vcat([Line[i][:] for i in rand_lines][:]...))
rand_tot = Int64.(unique([Int64.(sphere(center,128,10));hline;vline;rand_spirals;rand_lines]))

#%%
rand_brain = ReconstructBrain(brain,rand_tot)
figure()
imshow(rand_brain,cmap="gray")
println(sum((brain-brain_rcst).^2)/sum(brain.^2))
println(sum((brain-rand_brain).^2)/sum(brain.^2))


global psnrs = []
global ssims = []
global sizes = []
global mstars = []
for j = 1:200
    m = 128*128
    global psnrs
    global ssims
    global sizes
    global mstars
    rand_spirals = sample([1:15;19:27],16,replace=false)
    rand_spirals = unique(vcat([Spiral[i][:] for i in rand_spirals][:]...))

    rand_lines = sample([1:25 ;27:39],3,replace=false)
    rand_lines = unique(vcat([Line[i][:] for i in rand_lines][:]...))
    rand_tot = Int64.(unique([Int64.(sphere(center,128,10));hline;vline;rand_spirals;rand_lines]))
    g = [randn(m) for i in 1:16]
    wt = wavelet(WT.haar,WT.Lifting)
    println(length(rand_tot))
    mstars = [mstars;mstar(m,rand_tot,g,wt,1.4,1e-3)]
    sizes = [sizes;length(rand_tot)]
    rand_brain = ReconstructBrain(brain,rand_tot)
    psnrs = [psnrs;ImageQualityIndexes.psnr(rand_brain,brain)]
    ssims = [ssims;ssim(rand_brain,brain)]
end


function maxSizeSpiral(spirals,n_s,lines,n_l)
    res = Int64.(unique([Int64.(sphere(center,128,10));hline;vline]))
    res_bis = Int64.(unique([Int64.(sphere(center,128,10));hline;vline]))
    idx_s = []
    idx = 0
    for i = 1:n_s
        for j = 1:length(spirals)
            res_ = unique([res;spirals[j]])
            if length(res_) > length(res_bis)
                res_bis = Array(res_)
                idx = j
            end
        end
        res = Array(res_bis)
        idx_s = [idx_s;idx]
    end
    for i = 1:n_l
        for j = 1:length(lines)
            res_ = unique([res;lines[j]])
            if length(res_) > length(res_bis)
                res_bis = Array(res_)
                idx = j
            end
        end
        res = Array(res_bis)
        idx_s = [idx_s;idx]
    end
    return (res,idx_s)
end
max_spiral = 15
max_line = 0
(q,z) = maxSizeSpiral(Spiral,max_spiral,Line,max_line)
rand_spirals = z[1:max_spiral]
rand_lines = z[(max_spiral+1):end]
rand_spirals = unique(vcat([Spiral[i][:] for i in rand_spirals][:]...))
rand_lines = unique(vcat([Line[i][:] for i in rand_lines][:]...))
rand_tot = unique([Int64.(sphere(center,128,10));rand_spirals;rand_lines])
rand_brain = ReconstructBrain(brain,q)


#idx_spirals = [3;15;9;24;21;6;27;12;13;1;7;22;25;10;4]#16;10]
#idx_spirals = unique(vcat([Spiral[i][:] for i in idx_spirals][:]...))

#idx_lines = []#[8;2;33;24;19;27]#;1;10;17;28]#;29;18;35;21;34;6;13;7]
#idx_lines = unique(vcat([Line[i][:] for i in idx_lines][:]...))
#idx_tot = unique([Int64.(sphere(center,128,10));idx_spirals;idx_lines])
#hline = Int64.([coordToIdx(i,center+j,n) for i = 1:n for j =-1:3])
#vline = Int64.([coordToIdx(center+j,i,n) for i = 1:n for j =-1:3])
#idx = Int64.(unique([Int64(coordToIdx(center,center,n));vline;hline;idx_tot]))
#mst = 1.399846