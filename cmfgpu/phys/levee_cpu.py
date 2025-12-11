
import torch
import numpy as np

def levee_fldstg_cpu(
    total_storage: torch.Tensor,
    river_length: torch.Tensor,
    river_width: torch.Tensor,
    river_height: torch.Tensor,
    catchment_area: torch.Tensor,
    flood_depth_table: torch.Tensor, # (N, num_levels)
    levee_crown_height: torch.Tensor,
    levee_fraction: torch.Tensor,
    levee_base_height: torch.Tensor,
    levee_base_storage: torch.Tensor,
    num_flood_levels: int
):
    """
    CPU implementation of levee flood stage calculation for verification.
    Iterates over the batch and applies the logic for each element.
    """
    
    N = total_storage.shape[0]
    
    # Outputs
    river_storage = torch.zeros_like(total_storage)
    floodplain_storage = torch.zeros_like(total_storage)
    protected_storage = torch.zeros_like(total_storage)
    river_depth = torch.zeros_like(total_storage)
    floodplain_depth = torch.zeros_like(total_storage)
    protected_depth = torch.zeros_like(total_storage)
    floodplain_fraction = torch.zeros_like(total_storage)
    floodplain_area = torch.zeros_like(total_storage)
    
    # Convert to numpy for easier scalar processing in loop
    # (Assuming inputs are on CPU)
    total_storage_np = total_storage.numpy()
    river_length_np = river_length.numpy()
    river_width_np = river_width.numpy()
    river_height_np = river_height.numpy()
    catchment_area_np = catchment_area.numpy()
    flood_depth_table_np = flood_depth_table.numpy()
    levee_crown_height_np = levee_crown_height.numpy()
    levee_fraction_np = levee_fraction.numpy()
    levee_base_height_np = levee_base_height.numpy()
    levee_base_storage_np = levee_base_storage.numpy()
    
    for idx in range(N):
        # Extract single element parameters
        s_total = total_storage_np[idx]
        L = river_length_np[idx]
        W = river_width_np[idx]
        H = river_height_np[idx]
        A_catch = catchment_area_np[idx]
        depth_profile = flood_depth_table_np[idx]
        h_lev = levee_crown_height_np[idx]
        frac_lev = levee_fraction_np[idx]
        h_base = levee_base_height_np[idx]
        s_base = levee_base_storage_np[idx]
        
        # Derived parameters
        river_max_storage = L * W * H
        dwth_inc = (A_catch / L) / num_flood_levels
        levee_distance = frac_lev * (A_catch / L)
        
        # Calculate storage profile and gradient profile on the fly
        # (In Fortran/Python impl these were precomputed, but here we have depth table)
        storage_profile = np.zeros(num_flood_levels)
        gradient_profile = np.zeros(num_flood_levels)
        
        dsto_fil = river_max_storage
        dhgt_pre = 0.0
        
        for i in range(num_flood_levels):
            width_mid = W + dwth_inc * ((i + 1) - 0.5)
            height_diff = depth_profile[i] - dhgt_pre
            dsto_add = L * width_mid * height_diff
            storage_profile[i] = dsto_fil + dsto_add
            gradient_profile[i] = height_diff / dwth_inc
            dsto_fil = storage_profile[i]
            dhgt_pre = depth_profile[i]
            
        # Calculate levee_top_storage
        dhgt_dif = h_lev - h_base
        s_top = s_base + (levee_distance + W) * dhgt_dif * L
        
        # Calculate levee_fill_storage
        # [2] levee fill storage (water in both river side & protected side)
        i = 0
        dsto_fil = river_max_storage
        dwth_fil = W
        dhgt_pre = 0.0
        
        while i < num_flood_levels and h_lev > depth_profile[i]:
            dsto_fil = storage_profile[i]
            dwth_fil = dwth_fil + dwth_inc
            dhgt_pre = depth_profile[i]
            i += 1
            
        if i < num_flood_levels:
            dhgt_now = h_lev - dhgt_pre
            dwth_add = dhgt_now / gradient_profile[i]
            dsto_add = (dwth_add * 0.5 + dwth_fil) * dhgt_now * L
            s_fill = dsto_fil + dsto_add
        else:
            dhgt_now = h_lev - dhgt_pre
            dsto_add = dwth_fil * dhgt_now * L
            s_fill = dsto_fil + dsto_add

        # --- Logic Cases ---
        
        r_sto = 0.0
        f_sto = 0.0
        p_sto = 0.0
        r_dph = 0.0
        f_dph = 0.0
        p_dph = 0.0
        f_frc = 0.0
        
        if s_total > river_max_storage:
            # [Case-1] Under Levee Base
            if s_total < s_base:
                i = 0
                dsto_fil = river_max_storage
                dwth_fil = W
                ddph_fil = 0.0
                
                while i < num_flood_levels and s_total > storage_profile[i]:
                    dsto_fil = storage_profile[i]
                    dwth_fil = dwth_fil + dwth_inc
                    ddph_fil = ddph_fil + gradient_profile[i] * dwth_inc
                    i += 1
                
                if i < num_flood_levels:
                    dsto_add = s_total - dsto_fil
                    term = dwth_fil**2 + 2.0 * dsto_add / L / gradient_profile[i]
                    dwth_add = -dwth_fil + np.sqrt(term)
                    f_dph = ddph_fil + gradient_profile[i] * dwth_add
                else:
                    dsto_add = s_total - dsto_fil
                    dwth_add = 0.0
                    f_dph = ddph_fil + dsto_add / dwth_fil / L
                
                r_sto = river_max_storage + L * W * f_dph
                r_dph = r_sto / L / W
                f_sto = max(s_total - r_sto, 0.0)
                f_frc = (-W + dwth_fil + dwth_add) / (dwth_inc * num_flood_levels)
                f_frc = max(0.0, min(1.0, f_frc))
                
                p_sto = 0.0
                p_dph = 0.0

            # [Case-2] Under Levee Top
            elif s_total < s_top:
                dsto_add = s_total - s_base
                dwth_add = levee_distance + W
                f_dph = h_base + dsto_add / dwth_add / L
                
                r_sto = river_max_storage + L * W * f_dph
                r_dph = r_sto / L / W
                f_sto = max(s_total - r_sto, 0.0)
                f_frc = frac_lev
                
                p_sto = 0.0
                p_dph = 0.0

            # [Case-3] Filling Protected
            elif s_total < s_fill:
                f_dph = h_lev
                r_sto = river_max_storage + L * W * f_dph
                r_dph = r_sto / L / W
                
                f_sto = max(s_top - r_sto, 0.0)
                p_sto = max(s_total - r_sto - f_sto, 0.0)
                
                # Protected side stage calculation
                ilev = int(frac_lev * num_flood_levels)
                dsto_fil = s_top
                dwth_fil = 0.0
                ddph_fil = 0.0
                
                i = ilev
                while i < num_flood_levels:
                    dsto_add = (levee_distance + W) * (h_lev - depth_profile[i]) * L
                    if s_total < storage_profile[i] + dsto_add:
                        break
                    dsto_fil = storage_profile[i] + dsto_add
                    dwth_fil = dwth_inc * (i + 1) - levee_distance
                    ddph_fil = depth_profile[i] - h_base
                    i += 1
                
                if i < num_flood_levels:
                    dsto_add = s_total - dsto_fil
                    term = dwth_fil**2 + 2.0 * dsto_add / L / gradient_profile[i]
                    dwth_add = -dwth_fil + np.sqrt(term)
                    ddph_add = dwth_add * gradient_profile[i]
                    p_dph = h_base + ddph_fil + ddph_add
                    
                    f_frc = (dwth_fil + levee_distance) / (dwth_inc * num_flood_levels)
                    f_frc = max(0.0, min(1.0, f_frc))
                else:
                    dsto_add = s_total - dsto_fil
                    ddph_add = dsto_add / dwth_fil / L
                    p_dph = h_base + ddph_fil + ddph_add
                    f_frc = 1.0

            # [Case-4] Overtopped
            else:
                i = 0
                dsto_fil = river_max_storage
                dwth_fil = W
                ddph_fil = 0.0
                
                while i < num_flood_levels and s_total > storage_profile[i]:
                    dsto_fil = storage_profile[i]
                    dwth_fil = dwth_fil + dwth_inc
                    ddph_fil = ddph_fil + gradient_profile[i] * dwth_inc
                    i += 1
                
                if i < num_flood_levels:
                    dsto_add = s_total - dsto_fil
                    term = dwth_fil**2 + 2.0 * dsto_add / L / gradient_profile[i]
                    dwth_add = -dwth_fil + np.sqrt(term)
                    f_dph = ddph_fil + gradient_profile[i] * dwth_add
                else:
                    dsto_add = s_total - dsto_fil
                    dwth_add = 0.0
                    f_dph = ddph_fil + dsto_add / dwth_fil / L
                
                f_frc = (-W + dwth_fil + dwth_add) / (dwth_inc * num_flood_levels)
                
                r_sto = river_max_storage + L * W * f_dph
                r_dph = r_sto / L / W
                
                dsto_add = (f_dph - h_lev) * (levee_distance + W) * L
                f_sto = max(s_top + dsto_add - r_sto, 0.0)
                p_sto = max(s_total - r_sto - f_sto, 0.0)
                p_dph = f_dph

        else:
            # [Case-0] In River
            r_sto = s_total
            r_dph = s_total / L / W
            f_sto = 0.0
            f_dph = 0.0
            f_frc = 0.0
            p_sto = 0.0
            p_dph = 0.0
            
        # Store results
        river_storage[idx] = r_sto
        floodplain_storage[idx] = f_sto
        protected_storage[idx] = p_sto
        river_depth[idx] = r_dph
        floodplain_depth[idx] = f_dph
        protected_depth[idx] = p_dph
        floodplain_fraction[idx] = f_frc
        floodplain_area[idx] = f_frc * A_catch

    return (
        river_storage,
        floodplain_storage,
        protected_storage,
        river_depth,
        floodplain_depth,
        protected_depth,
        floodplain_fraction,
        floodplain_area
    )
