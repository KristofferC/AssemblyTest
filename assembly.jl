
module Foo
using FEMSparse
using FEMBase, Base.Threads
using JuliaFEM
using Tensors

struct Poisson <: FieldProblem end

include("coloring.jl")

FEMBase.get_unknown_field_name(::Problem{Poisson}) = "u"

function assemble_threaded!(problem::Problem{Poisson},
                  elements::Vector{T}, t) where {T <: Element}
    assemblers = [FEMSparse.start_assemble(problem.assembly.K_csc, problem.assembly.f_csc) for i in 1:Threads.nthreads()]
    local_buffers = [allocate_buffer(problem, elements) for i in 1:Threads.nthreads()]
    for (color, elements) in FEMBase.get_color_ranges(elements)
        Threads.@threads for i in 1:length(elements)
            element = elements[i]
            tid = Threads.threadid()
            assemble_element!(problem, assemblers[tid], local_buffers[tid], element, t)
        end
    end
end

function assemble_normal!(problem::Problem{Poisson},
                             elements::Vector{T}, t) where {T <: Element}
    assembler = FEMSparse.start_assemble(problem.assembly.K_csc, problem.assembly.f_csc)
    local_buffer = allocate_buffer(problem, elements)
    for element in elements
        assemble_element!(problem, assembler, local_buffer, element, t)
    end
end

function allocate_buffer(problem::Problem{Poisson}, ::Vector{Element{El}}) where El
    dim = get_unknown_field_dimension(problem)

    bi = BasisInfo(El)
    ndofs = length(bi)
    Ke = zeros(ndofs, ndofs)
    fe = zeros(ndofs)
    gdofs = zeros(Int, ndofs)
    nnodes = length(El)
    ndofs = dim*nnodes

    return PoissonLocalBuffer(Ke, fe, gdofs, bi, ndofs)
end

struct PoissonLocalBuffer{B <: BasisInfo, T}
    Ke::Matrix{T}
    fe::Vector{T}
    gdofs::Vector{Int}
    bi::B
    ndofs::Int
end

const XX = Vec{3}.([[-134.123, 44.1227, 160.297], [-132.033, 46.5369, 157.883], [-132.033, 43.6249, 157.883], [-135.466, 46.5369, 157.883], [-133.078,45.3298, 159.09], [-132.033, 45.0809, 157.883], [-133.078, 43.8738, 159.09], [-134.794, 45.3298, 159.09], [-133.749, 46.5369, 157.883], [-133.749, 45.0809, 157.883]])

function assemble_element!(problem::Problem{Poisson},
                            assembly::FEMSparse.AssemblerSparsityPattern,
                            buffer::PoissonLocalBuffer,
                            element::Element{E},
                            t::Float64) where E

    Ke, fe, gdofs, ndofs, bi = buffer.Ke, buffer.fe, buffer.gdofs, buffer.ndofs, buffer.bi
    cheating = true

    fill!(Ke, 0.0)
    fill!(fe, 0.0)
    has_coefficient = false #haskey(element, "coefficient")
    has_source = false #haskey(element, "source")
    if !cheating
        X = interpolate(element, "geometry", t)
    else
        X = XX
    end
    @inbounds for ip in 1:20 # get_integration_points(element)
        eval_basis!(bi, X, Vec(0.5,0.5,0.5))# (ip.coords))

        J, detJ, N, dN = bi.J, bi.detJ, bi.N, bi.grad

        dV = detJ * 0.3 #ip.weight
        k = 1.0
        if has_coefficient
            k = element("coefficient", ip, t)::Float64
        end
        for i in 1:ndofs
            for j in 1:ndofs
                Ke[j, i] += k * (dN[i] ⋅ dN[j]) * dV
            end
        end
        if has_source
            f = element("source", ip, time)::Vector{Float64}
            for i in 1:ndofs
                fe[i] += (N[i] ⋅ f[i]) * dV
            end
        end
    end
    FEMBase.get_gdofs!(gdofs, problem, element)
    @inbounds FEMSparse.assemble_local!(assembly, gdofs, Ke, fe)

    return nothing
end

mesh = abaqus_read_mesh("EIFFEL_TOWER_TET10_921317.inp")
#mesh = abaqus_read_mesh("EIFFEL_TOWER_TET10_220271.inp")

function test_assembler_timing()
    renumber_mesh!(mesh)
    tower = Problem(Poisson, "test problem", 1)
    tower_elements = create_elements(mesh, "TOWER")
    add_elements!(tower, tower_elements)
    coloring = JuliaFEM.create_coloring(mesh)
    FEMBase.assign_colors!(tower, coloring)
    tower.assembly.K_csc = FEMBase.create_sparsity_pattern!(tower)
    tower.assembly.f_csc = zeros(Float64, size(tower.assembly.K_csc, 2))
    t = 1.0
    initialize!(tower, t)

    elements = group_by_element_type(tower.elements)[Element{Tet10}]

    assemble_normal!(tower, elements, t)
    empty!(tower.assembly)
    assemble_threaded!(tower, elements, t)
    s_single = @elapsed @time assemble_normal!(tower, elements, t)
    empty!(tower.assembly)
    s_thread = @elapsed @time assemble_threaded!(tower, elements, t)
    println("Speedup: ", s_single / s_thread)
end


function renumber_mesh!(mesh)
    # Renumber nodes and elements so they start at 1

    #########
    # Nodes #
    #########
    node_mapping = Dict{Int, Int}()
    nodes = Dict{Int, Vector{Float64}}()
    for (i, node_id) in enumerate(sort(collect(keys(mesh.nodes))))
        nodes[i] = mesh.nodes[node_id]
        node_mapping[node_id] = i
    end

    ############
    # Elements #
    ############
    elements = Dict{Int, Vector{Float64}}()
    # The same comment about storing element numbers from 1:n applies here
    element_mapping = Dict{Int, Int}()
    for (i, element_id) in enumerate(sort(collect(keys(mesh.elements))))
        elements[i] = [node_mapping[z] for z in mesh.elements[element_id]]
        element_mapping[element_id] = i
    end

    ########
    # Sets #
    ########
    # The nodesets need use the new node ordering
    nodesets = Dict{Symbol, Set{Int}}()
    for (name, nodes) in mesh.node_sets
        nodesets[name] = Set(node_mapping[z] for z in nodes)
    end

    # So does the cell cets (element sets)
    elementsets = Dict{Symbol, Set{Int}}()
    for (name, elements) in mesh.element_sets
        elementsets[name] = Set(element_mapping[z] for z in elements)
    end

   # So does the cell cets (element sets)
    elementtypes = Dict{Int, Symbol}()
    for (element_id, typ) in mesh.element_types
        elementtypes[element_mapping[element_id]] = typ
    end

    surfacesets = Dict{Symbol, Vector{Tuple{Int, Symbol}}}()
    for (name, surface) in mesh.surface_sets
        surfacesets[name] = [(element_mapping[z[1]], z[2]) for z in surface]
    end

    mesh.nodes = nodes
    mesh.elements = elements
    mesh.node_sets = nodesets
    mesh.element_sets = elementsets
    mesh.element_types = elementtypes
    mesh.surface_sets = surfacesets
    return mesh
end

test_assembler_timing()

end