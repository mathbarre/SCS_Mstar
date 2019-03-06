using DelimitedFiles

function Spirals(dim::Int64,center::Int64,theta::Float64,rho::Float64,nb_pts::Int64)::Array{Int64,1}
    # generate nb_pts along a spiral 
    res = [center]
    abscisse = Int64(mod(center-1,(dim))+1)
    ordonnee = Int64(floor((center-1)/(dim)))
    n = 1
    while length(res) < nb_pts
        (x,y) = (Int64(floor(n*rho*cos(theta*n)+abscisse)),Int64(floor(n*rho*sin(theta*n)+ordonnee)))
        z = x+dim*y
        if !(z in res) &&  1 <= z && z <= dim^2
            res = [res;z]
        end
        n += 1
    end
    return res
end

function Lines(dim::Int64,center::Int64,radius::Int64,step::Float64,nb_pts::Int64)::Array{Int64,1}
    abscisse = Int64(mod(center-1,(dim))+1)
    ordonnee = Int64(floor((center-1)/(dim)))
    pt_on_cercle = (randn(2))
    pt_on_cercle /= sqrt(pt_on_cercle[1]^2 + pt_on_cercle[2]^2)
    pt_on_cercle *= radius
    pt_on_cercle = floor.(pt_on_cercle+[abscisse;ordonnee])
    x0 = pt_on_cercle - [abscisse;ordonnee]
    x0 /= sqrt(sum(x0.^2))
    (a,b) = (x0[2],-x0[1])
    res = [Int64(pt_on_cercle[1]+dim*pt_on_cercle[2])]
    sign_step = 1
    n=1
    limit = 0
    (x_,y_) = Int64.(pt_on_cercle)
    while length(res) < nb_pts && limit < 100
        (x,y) = (Int64(floor(x_+n*a*step*sign_step)),Int64(floor(y_+n*b*step*sign_step)))
        z = x+dim*y
        if !(z in res) &&  1 <= z && z <= dim^2
            res = [res;z]
        else 
            limit +=1
        end
        if (sign_step == 1)
            n +=1
        end
        sign_step *= -1
    end
    return res 
end
dim = 128
nb_pts = 300

img = zeros(dim,dim)

for theta = -0.2:0.05:0.2
    for rho = 0.1:0.05:0.2
        res = Spirals(dim,Int64(dim/2+dim/2*dim),theta,rho,nb_pts)
        if length(res) == nb_pts
            writedlm("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Spiral/Spiral_$(theta)_$(rho)_$(dim)_$(nb_pts).data",res)
            img[res] .= 1
        end
    end
end

imshow(img)
img = zeros(dim,dim)
nb_pts = 120
for i = 1:10
    for radius = [5 10 ]
        for step = [1.0 1.3]
            res = Lines(dim,Int64(dim/2+dim/2*dim),radius,step,nb_pts)
            if length(res) == nb_pts
                writedlm("/Users/mathieubarre/Documents/These/Mstar/juliaProg/Forms_128/Lines/Line_$(i)_$(radius)_$(step)_$(dim)_$(nb_pts).data",res)
                img[res] .= 1
            end
        end
    end
end
imshow(img)