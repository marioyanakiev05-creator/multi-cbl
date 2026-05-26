
############################################################
# RDKit
############################################################

const Chem = pyimport("rdkit.Chem")

############################################################
# FUNCTIONAL GROUPS
############################################################

#=
const FG_SMARTS = Dict(
    # Hydrocarbons
    "Alkane" => "[CX4]",  # sp3 carbon with 4 bonds
    "Alkene" => "[CX3]=[CX3]",  # C=C double bond
    "Alkyne" => "[CX2]#[CX2]",  # C≡C triple bond
    "Arene" => "c",  # Aromatic carbon
    
    # Halides
    "Chloroalkane" => "[CX4][Cl]",  # C-Cl (aliphatic)
    "Fluoroalkane" => "[CX4][F]",  # C-F (aliphatic)
    "Bromoalkane" => "[CX4][Br]",  # C-Br (aliphatic)
    "Iodoalkane" => "[CX4][I]",  # C-I (aliphatic)
    
    # Oxygen-containing groups
    "Acyl chloride" => "[CX3](=[OX1])[Cl]",  # R-C(=O)-Cl
    "Alcohol" => "[CX4][OX2H]",  # R-OH (aliphatic)
    "Aldehyde" => "[CX3H1](=O)[#6]",  # R-CHO
    "Ketone" => "[#6][CX3](=O)[#6]",  # R-C(=O)-R"
    "Carboxylic acid" => "[CX3](=O)[OX2H1]",  # R-COOH
    "Acid anhydride" => "[CX3](=[OX1])[OX2][CX3](=[OX1])",  # R-C(=O)-O-C(=O)-R"
    "Ester" => "[#6][CX3](=O)[OX2][#6]",  # R-C(=O)-O-R"
    "Ether" => "[OD2]([#6])[#6]",  # R-O-R" (not in C=O)
    "Phenol" => "[OX2H][c]",  # Ar-OH (aromatic)
    "Enol" => "[OX2H][CX3]=[CX3]",  # C=C-OH
    
    # Nitrogen-containing groups
    "Amine" => "[NX3;H2,H1;!\$(NC=O)]",  # R-NH2 or R2-NH (not amide)
    "Amide" => "[NX3][CX3](=[OX1])",  # R-C(=O)-N
    "Nitrile" => "[NX1]#[CX2]",  # R-C≡N
    "Imide" => "[CX3](=[OX1])[NX3][CX3](=[OX1])",  # O=C-NH-C=O
    "Imine" => "[CX3]=[NX2;!\$(N=O)]",  # R-C=N-R" (not nitro)
    
    # Sulfur-containing groups
    "Thiol" => "[#16X2H]",  # R-SH
    "Sulfide" => "[#16X2]([#6])[#6]",  # R-S-R"
# )

const FG_NAMES = sort(collect(keys(FG_SMARTS)))
const N_FG     = length(FG_NAMES)

const FG_PATTERNS = Dict(
    k => Chem.MolFromSmarts(v) for (k, v) in FG_SMARTS
)
=#

const FG_SMARTS = Dict(
    "Chlorine"          => "[CX4][Cl]",  
    "Ester Linkage"     => "[#6][CX3](=O)[OX2][#6]",
    "Aromatic Ring"     => "c1ccccc1",
    "Methyl Branch"     => "[CH3]",
    "Ethylene Backbone" => "[CH2]"
)
const FG_NAMES = sort(collect(keys(FG_SMARTS)))
const N_FG     = length(FG_NAMES)

const FG_PATTERNS = Dict(
    k => Chem.MolFromSmarts(v) for (k, v) in FG_SMARTS
)

############################################################
# FEATURIZE
############################################################

function featurize(smiles::AbstractString)::Union{Nothing, Vector{Float32}}
    mol = Chem.MolFromSmiles(smiles)
    pyisinstance(mol, Chem.rdchem.Mol) || return nothing

    feats = Vector{Float32}(undef, N_FG)
    for (i, k) in enumerate(FG_NAMES)
        n = length(mol.GetSubstructMatches(FG_PATTERNS[k]))
        feats[i] = n > 0 ? 1f0 : 0f0
    end
    return feats
end

############################################################
# NORMALIZE
############################################################

struct Normalizer
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function fit_normalizer(X::Matrix{Float32})
    μ = vec(mean(X, dims=2))
    σ = vec(std(X,  dims=2)) .+ 1f-6
    return Normalizer(μ, σ)
end

function apply_normalizer(norm::Normalizer, X::Matrix{Float32})
    return (X .- norm.μ) ./ norm.σ
end
