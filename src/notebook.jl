### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ b2eb6954-c1a6-11ee-05e9-f1129ba3294f
begin
	using Pkg
	cd("..")
	Pkg.activate(".")
end

# ╔═╡ 3f00e04a-af55-4d3b-9ad1-78df7b5a3ccd
begin
	using DataFrames, CSV
	using Turing, ForwardDiff
	using StatsPlots
	using LinearAlgebra 
end 

# ╔═╡ 49347b67-3702-4baf-97b2-8dabeaa8080d
begin
	matches = DataFrame(CSV.File("data/urc_matches_2022_2023.csv"))
end 

# ╔═╡ 7d84acab-19e2-4113-b746-f8094b33b835
md"""
Let's get the distribution of points scored:
"""

# ╔═╡ 5a265efa-5efe-47b4-8a87-01be3af839c9
density(matches.Diff, xlab = "Points Difference", ylab = "Frequency", title = "Distribution of Points Difference \nURC 2022/2023 Season", label = false)

# ╔═╡ 81020107-7883-4f4c-bf61-7a1d3a8175f2
begin
	density(matches.Home_Score, xlab = "Score", ylab = "Frequency", title = "Distribution of Points Difference \nURC 2022/2023 Season", label = "Home Team")
	density!(matches.Away_Score, title = "Distribution of Points Difference \nURC 2022/2023 Season", label = "Away Team")
end

# ╔═╡ 1774584a-017d-46d0-8b8f-adeac92eb834
begin
	scatter(matches.Home_Score, matches.Away_Score, xlab = "Home Score", ylab = "Away Score", xlim = (-1, 80), ylim = (-1, 80), label = false)
	plot!(0:80, 0:80, width = 2, color = :black, label = false)
end

# ╔═╡ 4aa0b3f3-b138-4f50-9367-58c22a7bc433
md"""
It appears there is a home team advantage. Let's quantify it further:
"""

# ╔═╡ 45736369-4450-4cd1-9799-af102e958d1a
begin
	bar(["Home Win %", "Away Win %", "Draw %"], [mean(matches.Home_Win), mean(matches.Away_Win), mean(matches.Draw)], label = false, ylab = " ", title = "Aggregate Performance")
end

# ╔═╡ a1c15df4-35b6-4c01-b8c3-1fd52b37b6b1
md"""
In addition, it appears there is a difference of ~ 1 converted try towards the home Team.
"""

# ╔═╡ 80463019-a43c-450b-9418-7d7c74034287
mean(matches.Diff)

# ╔═╡ 8d2cf0dc-e0a6-44db-af25-cfac9c0627b4
median(matches.Diff)

# ╔═╡ 245e4673-73bc-429a-999c-4150a639ec00
std(matches.Diff)

# ╔═╡ 667a4a82-89b0-46d1-b2f8-c1866449f03e
begin
	plot(matches.Match, matches.Diff, xlab = "Matches", label = false)
	hline!([mean(matches.Diff)], label = "Average Point Difference")
	hline!([mean(matches.Diff) + 1.96*std(matches.Diff)], label = "Upper 95% Point Difference")
	hline!([mean(matches.Diff) - 1.96*std(matches.Diff)], label = "Lower 95% Point Difference")
end

# ╔═╡ c039fb71-7e80-4034-929e-371df761898c
begin
	team_dict = Dict{String, Int64}()
	for (i, team) in enumerate(unique(matches.Home_Team))
		team_dict[team] = i
	end
end

# ╔═╡ 23ec80f3-3bb5-41f0-a996-4f047c16bdc8
team_dict

# ╔═╡ d46e802c-25b8-47f3-b7c6-2f5161b3bce0
@model function rugby_matches(home_teams, away_teams, diff, dict, ::Type{T} = Float64) where {T}
    # Hyper priors
    z = zeros(length(team_dict))
	σatt ~ Exponential(1)
    σdef ~ Exponential(1)
    μatt ~ MvNormal(z, 0.1 * I)
    μdef ~ MvNormal(z, 0.1 * I)
    
    home ~ Normal(0, 1)
	σ_home ~ truncated(Normal(0, 1); lower=0)
	σ_away ~ truncated(Normal(0, 1); lower=0)
	σ_univ ~ truncated(Normal(0, 1); lower=0)
        
    # Team-specific effects 
    att ~ MvNormal(μatt, σatt * I)
    def ~ MvNormal(μdef, σdef * I)
        
    # Zero-sum constrains
    offset = mean(att) + mean(def)
    
    θ_home = Vector{Real}(undef, length(home_teams))
    θ_away = Vector{Real}(undef, length(home_teams))
        
    # Modeling score-rate and scores (as many as there were games in the league)
    for i in 1:length(home_teams)
        # score-rate
        θ_home[i] = home + att[dict[home_teams[i]]] + def[dict[away_teams[i]]] - offset
        θ_away[i] = att[dict[away_teams[i]]] + def[dict[home_teams[i]]] - offset
	end

	# scores
	return diff ~ MvNormal(θ_home .- θ_away, σ_univ * I)
end

# ╔═╡ d55e888d-c485-492c-ac7d-cf18a4c40683
simulate_matches = rugby_matches(matches.Home_Team, matches.Away_Team, matches.Diff, team_dict)

# ╔═╡ 71cbf417-b1f2-4cee-a226-69d63c78b779
match_sample = Turing.sample(
	simulate_matches, 
	NUTS(0.6; adtype=AutoForwardDiff(; chunksize=0)),
	5_000; 
	discard_adapt=false)

# ╔═╡ c7502c64-bdd0-4607-acf3-505ead15fe47
describe(match_sample)

# ╔═╡ 7939e23c-648c-40b4-934a-c84a2f90bc33
begin
	post_att = collect(get(match_sample[2_500:end, :, :], :att)[1])
	post_def = collect(get(match_sample[2_500:end, :, :], :def)[1])
	post_home = collect(get(match_sample[2_500:end, :, :], :home)[1])
	global_sd = collect(get(match_sample[2_500:end, :, :], :σ_univ)[1])
end;

# ╔═╡ 9646442a-3352-4ea7-b6ff-57d079a51510
density(post_home)

# ╔═╡ aecb481f-41df-408c-ae92-8f66612e2f3f
begin
	teams_att = []
	teams_def = []
	for i in 1:length(post_att)
	    push!(teams_att, post_att[i])
	    push!(teams_def, post_def[i])
	end
end

# ╔═╡ b0384050-3bac-4cb8-abf0-6bc2bcbbfca4
begin
	density(teams_att[1], label = "Attack")
	density!(teams_def[1], label = "Defence")
end

# ╔═╡ db21f071-f239-45ef-83b9-0be1495313c3
function simulate_matches__(att₁, def₁, att₂, def₂, global_sd, home, n_matches; home_team = 1, zipped=true)
	home = mean(Array(home))
	att₁ = mean(Array(att₁))
	att₂ = mean(Array(att₂))
	def₁ = mean(Array(def₁))
	def₂ = mean(Array(def₂))
	global_σ = mean(Array(global_sd))
    if home_team == 1
        θ₁ = (home + att₁ + def₂) #> 0.0
        θ₂ = (att₂ + def₁) #> 0.0

    elseif home_team == 2
        θ₁ = (att₁ + def₂) > 0.0
        θ₂ = (home + att₂ + def₁) > 0.0
    else
        return DomainError(home_team, "Invalid home_team value")
    end
    
    diff = round.(rand(Normal(θ₁ - θ₂, global_σ), n_matches))
	match_status = Vector{Int8}(undef, n_matches)
	for i ∈ 1:n_matches
		if diff[i] > 0
			match_status[i] = 4
		elseif diff[i] < 0
			match_status[i] = 0
		else
			match_status[i] = 2
		end
	end

	if zipped == false
		DataFrame(
			"diff" => diff,
			"match_status" => match_status,
		)
	else
		[(s₁, s₂) for (s₁, s₂) in zip(scores₁, scores₂)]
	end
end


# ╔═╡ f9595003-159c-491d-b024-eeffd5f5c67d
match_sim = simulate_matches__(teams_att[1], teams_def[1], teams_att[2], teams_def[2], global_sd, post_home, 1_000_000; zipped=false)

# ╔═╡ 7263d113-c43f-42e4-9dc9-58e634d6720a
mean(match_sim.match_status)/4

# ╔═╡ 6307d3a0-c189-45cd-99ad-3f62702dd728
density(match_sim.diff)

# ╔═╡ 6cdcefb0-0e56-4b2f-a546-626e16fee088
mean(match_sim.diff)

# ╔═╡ 772c26ff-ea8d-40c2-9350-74f2b683ebec
team_list = unique(matches.Home_Team)

# ╔═╡ 76de30de-5eec-4391-93da-1c897a4ec959
match_sim_2 = simulate_matches__(teams_att[12], teams_def[12], teams_att[4], teams_def[4], global_sd, post_home, 1_000_000; zipped=false)

# ╔═╡ df201c22-042c-48ef-a378-8270f6d3c80c
density(match_sim_2.diff)

# ╔═╡ 140c0fa7-2744-4ee3-b1d5-28209890d11e
mean(match_sim_2.diff)

# ╔═╡ 45b8944f-6634-4877-b110-8f98f651924b
begin
	density(teams_att[12], label = "Attack")
	density!(teams_def[12], label = "Defence")
end

# ╔═╡ Cell order:
# ╠═b2eb6954-c1a6-11ee-05e9-f1129ba3294f
# ╠═3f00e04a-af55-4d3b-9ad1-78df7b5a3ccd
# ╠═49347b67-3702-4baf-97b2-8dabeaa8080d
# ╟─7d84acab-19e2-4113-b746-f8094b33b835
# ╠═5a265efa-5efe-47b4-8a87-01be3af839c9
# ╠═81020107-7883-4f4c-bf61-7a1d3a8175f2
# ╠═1774584a-017d-46d0-8b8f-adeac92eb834
# ╟─4aa0b3f3-b138-4f50-9367-58c22a7bc433
# ╠═45736369-4450-4cd1-9799-af102e958d1a
# ╟─a1c15df4-35b6-4c01-b8c3-1fd52b37b6b1
# ╠═80463019-a43c-450b-9418-7d7c74034287
# ╠═8d2cf0dc-e0a6-44db-af25-cfac9c0627b4
# ╠═245e4673-73bc-429a-999c-4150a639ec00
# ╠═667a4a82-89b0-46d1-b2f8-c1866449f03e
# ╠═c039fb71-7e80-4034-929e-371df761898c
# ╠═23ec80f3-3bb5-41f0-a996-4f047c16bdc8
# ╠═d46e802c-25b8-47f3-b7c6-2f5161b3bce0
# ╠═d55e888d-c485-492c-ac7d-cf18a4c40683
# ╠═71cbf417-b1f2-4cee-a226-69d63c78b779
# ╠═c7502c64-bdd0-4607-acf3-505ead15fe47
# ╠═7939e23c-648c-40b4-934a-c84a2f90bc33
# ╠═9646442a-3352-4ea7-b6ff-57d079a51510
# ╠═aecb481f-41df-408c-ae92-8f66612e2f3f
# ╠═b0384050-3bac-4cb8-abf0-6bc2bcbbfca4
# ╠═db21f071-f239-45ef-83b9-0be1495313c3
# ╠═f9595003-159c-491d-b024-eeffd5f5c67d
# ╠═7263d113-c43f-42e4-9dc9-58e634d6720a
# ╠═6307d3a0-c189-45cd-99ad-3f62702dd728
# ╠═6cdcefb0-0e56-4b2f-a546-626e16fee088
# ╠═772c26ff-ea8d-40c2-9350-74f2b683ebec
# ╠═76de30de-5eec-4391-93da-1c897a4ec959
# ╠═df201c22-042c-48ef-a378-8270f6d3c80c
# ╠═140c0fa7-2744-4ee3-b1d5-28209890d11e
# ╠═45b8944f-6634-4877-b110-8f98f651924b
