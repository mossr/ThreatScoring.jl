heading(a, b) = atan(b[2]-a[2], b[1]-a[1])
angle_to_target(xy, xy_target) = heading(xy, xy_target)

function headings(τ)
	H = [heading(τ[i], τ[i+1]) for i in 1:length(τ)-1]
	push!(H, H[end]) # copy final heading
	return H
end

circle(target::Target) = circle(target.xy, target.radius)
function circle(xy::Vector, r::Real)
	θ = LinRange(0, 2π, 500)
	return xy[1] .+ r*sin.(θ), xy[2] .+ r*cos.(θ)
end
