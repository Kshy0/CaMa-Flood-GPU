import triton
import triton.language as tl

@triton.jit
def compute_reservoir_outflow_kernel(
    catchment_idx_ptr,                      # *bool mask: 1 means river mouth 

    # river/flood variables
    river_inflow_ptr,                      # *f32 in/out river inflow
    river_outflow_ptr,                     # *f32 in/out river outflow
    river_storage_ptr,                     # *f32 in/out river storage
    flood_inflow_ptr,                      # *f32 in/out flood inflow
    flood_storage_ptr,                     # *f32 in/out flood storage

    # reservoir variables
    conservation_volume_ptr,            # *f32 conservation storage
    emergency_volume_ptr,               # *f32 emergency storage
    adjustment_volume_ptr,              # *f32 adjustment storage
    normal_outflow_ptr,                 # *f32 normal outflow
    adjustment_outflow_ptr,             # *f32 adjustment outflow
    flood_control_outflow_ptr,          # *f32 flood control outflow


    # other
    runoff_ptr,                             # *f32 runoff
    total_storage_ptr,
    outgoing_storage_ptr,                   # *f32 output for storage 
    time_step,                              # f32 scalar time step
    num_catchments,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    catchment_idx = tl.load(catchment_idx_ptr + offs, mask=mask, other=0)
    river_inflow = tl.load(river_inflow_ptr + catchment_idx, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + catchment_idx, mask=mask, other=0.0)
    river_storage = tl.load(river_storage_ptr + catchment_idx, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + catchment_idx, mask=mask, other=0.0)

    runoff = tl.load(runoff_ptr + catchment_idx, mask=mask, other=0.0)
    reservoir_inflow = river_inflow + flood_inflow + runoff

    total_storage = river_storage + flood_storage
    conservation_volume = tl.load(conservation_volume_ptr + offs, mask=mask, other=0.0)
    emergency_volume = tl.load(emergency_volume_ptr + offs, mask=mask, other=0.0)
    adjustment_volume = tl.load(adjustment_volume_ptr + offs, mask=mask, other=0.0)
    normal_outflow = tl.load(normal_outflow_ptr + offs, mask=mask, other=0.0)
    adjustment_outflow = tl.load(adjustment_outflow_ptr + offs, mask=mask, other=0.0)
    flood_control_outflow = tl.load(flood_control_outflow_ptr + offs, mask=mask, other=0.0)

    reservoir_outflow = tl.zeros_like(total_storage)


    # case 1: water use
    cond1 = total_storage <= conservation_volume
    reservoir_outflow = tl.where(
        cond1,
        normal_outflow * tl.sqrt(total_storage / conservation_volume),
        reservoir_outflow
    )

    # case 2: just above conservation volume
    cond2 = (total_storage > conservation_volume) & (total_storage <= adjustment_volume)
    reservoir_outflow = tl.where(
        cond2,
        normal_outflow + tl.exp(3.0 * tl.log((total_storage - conservation_volume) / (adjustment_volume - conservation_volume))) * (adjustment_outflow - normal_outflow),
        reservoir_outflow
    )

    # case 3: above adjustment volume, below emergency volume
    cond3 = (total_storage > adjustment_volume) & (total_storage <= emergency_volume)
    flood_period = reservoir_inflow >= flood_control_outflow

    # flood period logic
    outflow_flood = normal_outflow + ((total_storage - conservation_volume) / (emergency_volume - conservation_volume)) * (reservoir_inflow - normal_outflow)
    outflow_tmp = adjustment_outflow + tl.exp(0.1 * tl.log((total_storage - adjustment_volume) / (emergency_volume - adjustment_volume))) * (flood_control_outflow - adjustment_outflow)
    outflow_combined = tl.maximum(outflow_flood, outflow_tmp)

    # non-flood period logic
    outflow_nonflood = adjustment_outflow + tl.exp(0.1 * tl.log((total_storage - adjustment_volume) / (emergency_volume - adjustment_volume))) * (flood_control_outflow - adjustment_outflow)

    reservoir_outflow = tl.where(
        cond3 & flood_period,
        outflow_combined,
        reservoir_outflow
    )
    reservoir_outflow = tl.where(
        cond3 & ~flood_period,
        outflow_nonflood,
        reservoir_outflow
    )

    # case 4: emergency (above emergency volume)
    cond4 = total_storage > emergency_volume
    outflow_emergency = tl.where(
        reservoir_inflow >= flood_control_outflow,
        reservoir_inflow,
        flood_control_outflow
    )
    reservoir_outflow = tl.where(
        cond4,
        outflow_emergency,
        reservoir_outflow
    )

    reservoir_outflow = tl.clamp(reservoir_outflow, 0.0, total_storage / time_step)
    tl.store(river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask)
    tl.store(total_storage_ptr + catchment_idx, total_storage, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + offs, reservoir_outflow * time_step, mask=mask)


@triton.jit
def compute_manning_outflow_kernel(
    # Indices
    catchment_idx_ptr,                          # *i32: Catchment indices
    downstream_idx_ptr,                         # *i32: Downstream indices

    # river variables
    river_outflow_ptr,                      # *f32 in/out river outflow
    river_manning_ptr,                      # *f32 river Manning coefficient
    river_depth_ptr,                        # *f32 river depth
    river_width_ptr,                        # *f32 river width
    river_length_ptr,                       # *f32 river length
    river_storage_ptr,                      # *f32 river storage

    # flood variables
    flood_outflow_ptr,                      # *f32 in/out flood outflow
    flood_manning_ptr,                      # *f32 flood Manning coefficient
    flood_depth_ptr,                        # *f32 flood depth
    catchment_elevation_ptr,                # *f32 catchment ground elevation
    downstream_distance_ptr,                # *f32 distance to downstream unit
    flood_storage_ptr,                      # *f32 flood storage

    # other 
    total_storage_ptr,                      # *f32 total storage (in/out)
    outgoing_storage_ptr,                   # *f32 output for storage (fused part)
    time_step,                              # f32 scalar time step
    min_slope: tl.constexpr,                    # f32 min slope
    num_catchments,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    
    # Update upstream outflow
    catchment_idx = tl.load(catchment_idx_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    downstream_distance = tl.load(downstream_distance_ptr + catchment_idx, mask=mask, other=0.0)
    river_manning = tl.load(river_manning_ptr + catchment_idx, mask=mask, other=0.0)
    flood_manning = tl.load(flood_manning_ptr + catchment_idx, mask=mask, other=0.0)
    river_width = tl.load(river_width_ptr + catchment_idx, mask=mask, other=0.0)
    river_depth = tl.load(river_depth_ptr + catchment_idx, mask=mask, other=0.0)
    river_length = tl.load(river_length_ptr + catchment_idx, mask=mask, other=0.0)
    flood_depth = tl.load(flood_depth_ptr + catchment_idx, mask=mask, other=0.0)
    river_storage = tl.load(river_storage_ptr + catchment_idx, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + catchment_idx, mask=mask, other=0.0)

    catchment_elevation = tl.load(catchment_elevation_ptr + catchment_idx, mask=mask, other=0.0)
    catchment_elevation_downstream = tl.load(catchment_elevation_ptr + downstream_idx, mask=mask, other=0.0)
    river_slope = (catchment_elevation - catchment_elevation_downstream) / downstream_distance
    river_slope = tl.maximum(river_slope, min_slope)
    
    river_velocity = tl.sqrt(river_slope) * tl.exp((2.0/3.0) * tl.log(river_depth)) / river_manning
    river_outflow = river_velocity * river_depth * river_width
    flood_slope = tl.minimum(river_slope, 0.005)
    flood_velocity = tl.sqrt(flood_slope) * tl.exp((2.0/3.0) * tl.log(flood_depth) / flood_manning)
    flood_outflow = flood_velocity * tl.maximum(flood_storage / river_length - flood_depth * river_width, 0.0)
    tl.store(river_outflow_ptr + catchment_idx, river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + catchment_idx, flood_outflow, mask=mask)
    tl.store(total_storage_ptr + catchment_idx, river_storage + flood_storage, mask=mask)

    pos = tl.maximum(river_outflow, 0.0) + tl.maximum(flood_outflow, 0.0)
    neg = tl.minimum(river_outflow, 0.0) + tl.minimum(flood_outflow, 0.0)


    tl.atomic_add(outgoing_storage_ptr + offs, pos * time_step, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, -neg * time_step, mask=mask)



def compute_reservoir_outflow(
    reservoir_catchment_idx_ptr,             # *i32 index: catchment indices
    reservoir_upstream_idx_ptr,              # *i32 index: upstream indices for Manning kernel

    # River/flood variables
    river_inflow_ptr,                        # *f32 river inflow
    river_outflow_ptr,                       # *f32 river outflow
    river_storage_ptr,                       # *f32 river storage
    flood_inflow_ptr,                        # *f32 flood inflow
    flood_outflow_ptr,                       # *f32 flood outflow
    flood_storage_ptr,                       # *f32 flood storage

    # Reservoir parameters
    conservation_volume_ptr,                 # *f32 conservation storage
    emergency_volume_ptr,                    # *f32 emergency storage
    adjustment_volume_ptr,                   # *f32 adjustment storage
    normal_outflow_ptr,                      # *f32 normal outflow
    adjustment_outflow_ptr,                  # *f32 adjustment outflow
    flood_control_outflow_ptr,               # *f32 flood control outflow

    # Manning parameters
    river_manning_ptr,                       # *f32 river Manning coefficient
    river_depth_ptr,                         # *f32 river depth
    river_width_ptr,                         # *f32 river width
    river_length_ptr,                        # *f32 river length
    flood_manning_ptr,                       # *f32 flood Manning coefficient
    flood_depth_ptr,                         # *f32 flood depth

    # Elevation and distances
    catchment_elevation_ptr,                 # *f32 catchment ground elevation
    downstream_distance_ptr,                 # *f32 distance to downstream unit

    # Other
    runoff_ptr,                              # *f32 runoff
    total_storage_ptr,                       # *f32 total storage
    outgoing_storage_ptr,                    # *f32 storage output
    time_step,                               # f32 scalar time step
    num_reservoirs,                          # int total reservoirs
    num_upstreams,                           # int total upstream catchments
    min_slope,                               # f32 minimum slope
    BLOCK_SIZE
):

    reservoir_grid = lambda meta: (triton.cdiv(num_reservoirs, meta['BLOCK_SIZE']),)

    compute_reservoir_outflow_kernel[reservoir_grid](
        catchment_idx_ptr=reservoir_catchment_idx_ptr,
        river_inflow_ptr=river_inflow_ptr,
        river_outflow_ptr=river_outflow_ptr,
        river_storage_ptr=river_storage_ptr,
        flood_inflow_ptr=flood_inflow_ptr,
        flood_storage_ptr=flood_storage_ptr,
        conservation_volume_ptr=conservation_volume_ptr,
        emergency_volume_ptr=emergency_volume_ptr,
        adjustment_volume_ptr=adjustment_volume_ptr,
        normal_outflow_ptr=normal_outflow_ptr,
        adjustment_outflow_ptr=adjustment_outflow_ptr,
        flood_control_outflow_ptr=flood_control_outflow_ptr,
        runoff_ptr=runoff_ptr,
        total_storage_ptr=total_storage_ptr,
        outgoing_storage_ptr=outgoing_storage_ptr,
        time_step=time_step,
        num_catchments=num_reservoirs,
        BLOCK_SIZE=BLOCK_SIZE
    )

    upstream_grid = lambda meta: (triton.cdiv(num_upstreams, meta['BLOCK_SIZE']),)

    compute_manning_outflow_kernel[upstream_grid](
        catchment_idx_ptr=reservoir_upstream_idx_ptr,
        downstream_idx_ptr=reservoir_catchment_idx_ptr,
        river_outflow_ptr=river_outflow_ptr,
        river_manning_ptr=river_manning_ptr,
        river_depth_ptr=river_depth_ptr,
        river_width_ptr=river_width_ptr,
        river_length_ptr=river_length_ptr,
        river_storage_ptr=river_storage_ptr,
        flood_outflow_ptr=flood_outflow_ptr,
        flood_manning_ptr=flood_manning_ptr,
        flood_depth_ptr=flood_depth_ptr,
        catchment_elevation_ptr=catchment_elevation_ptr,
        downstream_distance_ptr=downstream_distance_ptr,
        flood_storage_ptr=flood_storage_ptr,
        total_storage_ptr=total_storage_ptr,
        outgoing_storage_ptr=outgoing_storage_ptr,
        time_step=time_step,
        min_slope=min_slope,
        num_catchments=num_upstreams,
        BLOCK_SIZE=BLOCK_SIZE
    )