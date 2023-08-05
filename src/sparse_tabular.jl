struct ModifiedSparseTabular <: POMDP{Int,Int,Int}
    T::Vector{SparseMatrixCSC{Float64, Int64}} # T[a][sp, s]
    R::Array{Float64, 2} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    isterminal::SparseVector{Bool, Int}
    initialstate::SparseVector{Float64, Int}
    discount::Float64
end

function ModifiedSparseTabular(pomdp::POMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    terminal = _vectorized_terminal(pomdp, S)
    T = POMDPTools.ModelTools.transition_matrix_a_s_sp(pomdp)
    R = _tabular_rewards(pomdp, S, A)
    O = POMDPTools.ModelTools.observation_matrix_a_sp_o(pomdp)
    b0 = _vectorized_initialstate(pomdp, S)
    return ModifiedSparseTabular(T,R,O,terminal,b0,discount(pomdp))
end

function _tabular_rewards(pomdp, S, A, terminal)
    R = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx, s) ∈ enumerate(S)
        if terminal[s_idx]
            R[s_idx, :] .= 0.0
            continue
        end
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx] = reward(pomdp, s, a)
        end
    end
    R
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(term,S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S, b0_vec)
        b0_vec[i] = pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

POMDPTools.ordered_states(pomdp::ModifiedSparseTabular) = axes(pomdp.R, 1)
POMDPs.states(pomdp::ModifiedSparseTabular) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::ModifiedSparseTabular) = eachindex(pomdp.T)
POMDPs.actions(pomdp::ModifiedSparseTabular) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::ModifiedSparseTabular) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::ModifiedSparseTabular) = ordered_observations(pomdp)

POMDPs.discount(pomdp::ModifiedSparseTabular) = pomdp.discount
POMDPs.initialstate(pomdp::ModifiedSparseTabular) = pomdp.initialstate
POMDPs.isterminal(pomdp::ModifiedSparseTabular, s::Int) = pomdp.isterminal[s]

n_states(pomdp::ModifiedSparseTabular) = length(states(pomdp))
n_actions(pomdp::ModifiedSparseTabular) = length(actions(pomdp))
n_observations(pomdp::ModifiedSparseTabular) = length(observations(pomdp))
