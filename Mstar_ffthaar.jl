@everywhere include("/Users/mathieubarre/Documents/mstar/withImage/scs_like/scs_julia.jl")

@everywhere using FFTW
@everywhere using StatsBase
@everywhere using Wavelets


@everywhere FFTW.set_num_threads(4)
using Distributed




@everywhere function myFFT(x::Array{Float64,1},s::String,idx_r::Array{Int64,1},idx_i::Array{Int64,1},m::Int64,wt::GLS{Wavelets.WT.PerBoundary})::Array{Float64,1}
    n_r = size(idx_r,1)
    n_i = size(idx_i,1)
    n = size(x,1)
    if s == "NT"
        x_r = zeros(m)
        #x_r[idx_r] = x[2:(Int((n+1)/2))]
        x_r[idx_r] = x[2:(n_r+1)]
        x_i = zeros(m)
        #x_i[idx_i] = x[Int((n+1)/2+1):end]
        x_i[idx_i] = x[(n_r+2):end]
        dr = real(ifftshift(ifft(fftshift(reshape(x_r,Int(sqrt(m)),Int(sqrt(m)))))))*sqrt(m)
        #dr = real((ifft((reshape(x_r,Int(sqrt(m)),Int(sqrt(m)))))))*m
        #dr = dr[:]
        di = -imag(ifftshift(ifft(fftshift(reshape(x_i,Int(sqrt(m)),Int(sqrt(m)))))))*sqrt(m)
        #di = -imag((ifft((reshape(x_i,Int(sqrt(m)),Int(sqrt(m)))))))*m
        #di = di[:]
        d = dwt(dr+di,wt)
        d = d[:]
        y = [-x[1]*ones(size(d,1))+d;-x[1]*ones(size(d,1))-d]
    else
        d1 = fftshift(fft(ifftshift(idwt(reshape(x[1:(Int(n/2))],Int(sqrt(m)),Int(sqrt(m))),wt))))/sqrt(m)
        #d1 = (fft((reshape(x[1:(Int(n/2))],Int(sqrt(m)),Int(sqrt(m))))))
        d1 =d1[:]
        #d1 = d1[idx_r]
        d1_r = real(d1[idx_r]) 
        d1_i = imag(d1[idx_i])
        d1 = [d1_r;d1_i]
        d2 = fftshift(fft(ifftshift(idwt(reshape(x[Int(n/2+1):end],Int(sqrt(m)),Int(sqrt(m))),wt))))/sqrt(m)
        #d2 = (fft((reshape(x[Int(n/2+1):end],Int(sqrt(m)),Int(sqrt(m))))))
        d2 =d2[:]
        #d2 = d2[idx_r]
        #d2 = [real(d2);imag(d2)]
        d2_r = real(d2[idx_r]) 
        d2_i = imag(d2[idx_i])
        d2 = [d2_r;d2_i]
        y = [-sum(x);d1-d2];
    end
    return y
end

function getDualIndex(n::Int64,i::Int64)::Int64
    blk_idx = ceil(i/sqrt(n))
    inter_idx = mod(i-1,sqrt(n))+1
    if blk_idx == 1 || blk_idx == sqrt(n)/2+1
       if inter_idx == 1 || inter_idx == sqrt(n)/2+1 
           idx = -1
       else
            idx = (blk_idx-1)*sqrt(n) + sqrt(n)+2-inter_idx
       end
    else
        blk_idx_sym = sqrt(n)+2-blk_idx
        if inter_idx == 1 || inter_idx == sqrt(n)/2+1 
           idx = (blk_idx_sym-1)*sqrt(n)+inter_idx
       else
           idx = (blk_idx_sym-1)*sqrt(n)+ sqrt(n)+2-inter_idx
       end 
    end
    return Int(idx)
end

function cleanIndex(idx::Array{Int64,1},m::Int64)::Tuple{Array{Int64,1},Array{Int64,1}}
    n = length(idx)
    idx_r = falses(m)
    idx_i = falses(m)
    idx_r[idx] = trues(n)
    idx_i[idx] = trues(n)
    for i in idx
        dual = getDualIndex(m,i)
        if dual != -1 && idx_r[dual]
            idx_r[i] = false 
            idx_i[i] = false
        end
        if dual == -1
            idx_i[i] = false
        end
    end
    return ((1:m)[idx_r],(1:m)[idx_i])
end

function orthoNull(idx::Array{Int64,1},m::Int64)::Tuple{Array{Bool,1},Array{Bool,1}}
    usefull = trues(m)
    scale = falses(m)
    usefull[idx] = falses(size(idx,1))
    for i in idx
        dual = getDualIndex(m,i)
        if dual != -1
            usefull[dual] = false
        end
    end
    scale[usefull] = trues(sum(usefull))
    for (i,b) in enumerate(usefull)
        if b
            dual = getDualIndex(m,i)
            if dual != -1 && usefull[dual]
                usefull[i] = false
            end
            if dual == -1
                scale[i] = false
            end
        end
    end
    return usefull,scale
end

function Fg(g::Array{Float64,1},usefull::Array{Bool,1},scale::Array{Bool,1},s::String,wt::GLS{Wavelets.WT.PerBoundary})::Array{Float64,1}
    n = size(g,1)
    m = size(usefull,1)
    if s == "NT"
        x_r = zeros(m)
        x_r[usefull] = g[1:Int(n/2)]
        x_r[scale] *= sqrt(2)
        x_i = zeros(m)
        x_i[usefull] = g[Int(n/2+1):end]
        x_i[scale] *= sqrt(2)
        dr = real(ifftshift(ifft(fftshift(reshape(x_r,Int(sqrt(m)),Int(sqrt(m)))))))*sqrt(m)
        #dr = real((ifft((reshape(x_r,Int(sqrt(m)),Int(sqrt(m)))))))*m
        #dr = dr[:]
        di = -imag(ifftshift(ifft(fftshift(reshape(x_i,Int(sqrt(m)),Int(sqrt(m)))))))*sqrt(m)
        #di = -imag((ifft((reshape(x_i,Int(sqrt(m)),Int(sqrt(m)))))))*m
        #di = di[:]
        d = dwt(dr+di,wt)
    else
        d1 = fftshift(fft(ifftshift(idwt(reshape(g,Int(sqrt(m)),Int(sqrt(m))),wt))))/sqrt(m)
        #d1 = (fft((reshape(x[1:(Int(n/2))],Int(sqrt(m)),Int(sqrt(m))))))
        d1 =d1[:]
        d1[scale]*= sqrt(2)
        d1 = d1[usefull]
        d = [real(d1);imag(d1)]
        
    end
    return d[:]
end

function mstar(m::Int64,idx::Array{Int64,1},g::Array{Array{Float64,1},1})::Float64
    (idx_r,idx_i) = cleanIndex(idx,m)
    n_r = length(idx_r)
    n_i = length(idx_i)
    usefull,scale = orthoNull(idx,m)
    sigma = 1.0/sqrt(sum(scale)/2+sum(usefull))
    rho = 1
    c = [1;zeros(n_r+n_i)]
    E = [1/sqrt(2*m);ones(n_r+n_i)]
    r_idx = falses(m)
    i_idx = falses(m)
    r_idx[idx_r] = trues(n_r)
    i_idx[idx_i] = trues(n_i)
    j=2
    for i = 1:m
        if r_idx[i]
            if !i_idx[i]
                E[j] = 1/sqrt(2)
            end
            j += 1
        end
    end
    wt = wavelet(WT.haar,WT.Lifting)
    @everywhere FFTspace(x,s) = myFFT(x,s,$idx_r,$idx_i,$m,$wt)
    iter = length(g)
    bs = [Fg(g[i][1:(2*sum(usefull))],usefull,scale,"NT",wt) for i in 1:iter]

    @time x = @distributed (+) for i in 1:iter
        SCSsolve(FFTspace,c,[bs[i];-bs[i]],1.4,1e-3,1e-3,1e-3,sigma = sigma,E=E)
    end
    return x/iter
end

function mstar_noNull(m::Int64,idx::Array{Int64,1},g::Array{Array{Float64,1},1})::Float64
    (idx_r,idx_i) = cleanIndex(idx,m)
    n = length(idx)
    n_r = length(idx_r)
    n_i = length(idx_i)
    #usefull,scale = orthoNull(idx,m)
    sigma = 1.0/sqrt(n)
    rho = 1
    c = [1;zeros(n_r+n_i)]
    E = [1/sqrt(2*m);ones(n_r+n_i)]
    r_idx = falses(m)
    i_idx = falses(m)
    r_idx[idx_r] = trues(n_r)
    i_idx[idx_i] = trues(n_i)
    j=2
    for i = 1:m
        if r_idx[i]
            if !i_idx[i]
                E[j] = 1/sqrt(2)
            end
            j += 1
        end
    end
    wt = wavelet(WT.haar,WT.Lifting)
    @everywhere FFTspace(x,s) = myFFT(x,s,$idx_r,$idx_i,$m,$wt)
    iter = length(g)
    

    @time x = @distributed (+) for i in 1:iter
        SCSsolve(FFTspace,c,[g[i];-g[i]],1.4,1e-3,1e-3,1e-3,sigma = sigma,E=E)
    end
    return x/iter
end
