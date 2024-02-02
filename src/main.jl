using DataFrames, CSV
using Turing, ForwardDiff
using StatsPlots
using LinearAlgebra  

println("Getting matches...")
matches = DataFrame(CSV.File("data/urc_matches_2022_2023.csv"))

function team_dict(m)
  teams = Dict{String, Int64}()
  for (i, team) in enumerate(unique(m.Home_Team))
    teams[team] = i
  end
  return teams
end

@model function rugby_matches(home_teams, away_teams, diff, dict, ::Type{T} = Float64) where {T}
  # Hyper priors
  z = zeros(length(dict))
  σatt ~ Exponential(1)
  σdef ~ Exponential(1)
  μatt ~ MvNormal(z, 0.1 * I)
  μdef ~ MvNormal(z, 0.1 * I)

  home ~ Normal(0, 1)
  σ_home ~ truncated(Normal(0, 1); lower=0.01)
  σ_away ~ truncated(Normal(0, 1); lower=0.01)
  σ_univ ~ truncated(Normal(0, 1); lower=0.01)

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

simulate_matches = rugby_matches(matches.Home_Team, matches.Away_Team, matches.Diff, team_dict(matches))

println("Estimating posteriors...")
num_samples = 5_000
half_samples = Int(num_samples/2)
match_sample = Turing.sample(
			     simulate_matches, 
			     NUTS(0.45; adtype=AutoForwardDiff(; chunksize=0)),
			     num_samples; 
			     discard_adapt=false);


println("Fetching posteriors...")
begin
  post_att = collect(get(match_sample[half_samples:end, :, :], :att)[1])
  post_def = collect(get(match_sample[half_samples:end, :, :], :def)[1])
  post_home = collect(get(match_sample[half_samples:end, :, :], :home)[1])
  global_sd = collect(get(match_sample[half_samples:end, :, :], :σ_univ)[1])
  teams_att = []
  teams_def = []

  for i in 1:length(post_att)
    push!(teams_att, post_att[i])
    push!(teams_def, post_def[i])
  end
end

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

function make_predictions(m, n)
  pred_diff = Vector{Float64}(undef, size(m)[1])
  team_dict_ = team_dict(m)
  for i ∈ 1:size(m)[1]
    get_home = m[i, :Home_Team]
    get_away = m[i, :Away_Team]
    pred_diff[i] =  mean(
			 simulate_matches__(
					    teams_att[team_dict_[get_home]], 
					    teams_def[team_dict_[get_home]], 
					    teams_att[team_dict_[get_away]], 
					    teams_def[team_dict_[get_away]], 
					    global_sd, 
					    post_home, 
					    n; 
					    zipped=false).diff)
  end
  return round.(pred_diff)
end

function make_league_table(m; prediction=false, n=100)
  matches = m[m.Round .!= "QF" .&& m.Round .!= "SF" .&& m.Round .!= "F", :]
  if prediction == true
    matches.Diff = make_predictions(matches, n)
  end
  league_table = DataFrame(
			   Position=[],
			   Team=[],
			   Played=[],
			   Won=[],
			   Loss=[],
			   Draw=[],
			   PD=[],
			   Points=[]
			   )
  position_ = 0
  for team_ ∈ unique(matches.Home_Team)
    played_ = size(matches[matches.Home_Team .== team_, :])[1]
    won_ = size(matches[matches.Home_Team .== team_ .&& matches.Diff .> 0, :])[1]
    loss_ = size(matches[matches.Home_Team .== team_ .&& matches.Diff .< 0, :])[1]
    draw_ = size(matches[matches.Home_Team .== team_ .&& matches.Diff .== 0, :])[1]
    gd_ = sum(matches[matches.Home_Team .== team_, :Diff])

    played_ = played_ + size(matches[matches.Away_Team .== team_, :])[1]
    won_ = won_ + size(matches[matches.Away_Team .== team_ .&& matches.Diff .< 0, :])[1]
    loss_ = loss_ + size(matches[matches.Away_Team .== team_ .&& matches.Diff .> 0, :])[1]
    draw_ = draw_ + size(matches[matches.Away_Team .== team_ .&& matches.Diff .== 0, :])[1]
    gd_ = gd_ - sum(matches[matches.Away_Team .== team_, :Diff])

    position_ = position_ + 1
    points_ = 4 * won_ + 2 * draw_
    table_line = [position_, team_, played_, won_, loss_, draw_, Int(gd_), points_]
    push!(league_table, table_line)
  end
  sort!(league_table, [:Points, :PD], rev = true)
  league_table.Position = 1:size(league_table)[1]
  return league_table
end

println("Simulating league...")
print(make_league_table(matches; prediction=true, n=1_000_000))

println("\n")
println("Retrieving actual league table...")
print(make_league_table(matches))

