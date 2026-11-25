// HYDROFORGE METAL KERNEL BODY: compute_downscale_flood_depth
    long num_pixels = *args.num_pixels;
    long ensemble_size = *args.ensemble_size;
    long total = num_pixels * ensemble_size;
    if ((long)i >= total) return;

    long p, member;
    cmf_decompose_index(i, num_pixels, ensemble_size, p, member);
    long cat_off = member * *args.num_catchments;
    long pix_off = member * num_pixels;

    int ci = args.pixel_catchment_idx_ptr[p];
    float flddif = args.pixel_flood_diff_ptr[(batched_pixel_flood_diff ? pix_off : 0) + p];
    float depth = 0.0f;

    if (HAS_LEVEE) {
        bool is_prot = args.pixel_is_protected_ptr[(batched_pixel_is_protected ? pix_off : 0) + p] != 0;
        if (is_prot) {
            float levdph = args.protected_depth_ptr[(batched_protected_depth ? cat_off : 0) + ci];
            if (levdph > flddif) {
                depth = (levdph - flddif) + *args.protected_depth_offset;
            }
        } else {
            float flddph = args.flood_depth_ptr[(batched_flood_depth ? cat_off : 0) + ci];
            if (flddph > flddif) {
                depth = flddph - flddif;
            }
        }
    } else {
        float flddph = args.flood_depth_ptr[(batched_flood_depth ? cat_off : 0) + ci];
        if (flddph > flddif) {
            depth = flddph - flddif;
        }
    }

    if (!HAS_TRIBUTARY) {
        float perm_depth = *args.permanent_water_depth;
        bool is_perm = args.pixel_is_permanent_water_ptr[(batched_pixel_is_permanent_water ? pix_off : 0) + p] != 0;
        if (perm_depth > 0.0f && is_perm) {
            depth = max(perm_depth, depth);
        }
    }

    args.pixel_flood_depth_ptr[pix_off + p] = depth;

// HYDROFORGE METAL KERNEL BODY: compute_downscale_tributary_depth
    long num_trib = *args.num_tributary_pixels;
    long ensemble_size = *args.ensemble_size;
    long total = num_trib * ensemble_size;
    if ((long)i >= total) return;

    long t, member;
    cmf_decompose_index(i, num_trib, ensemble_size, t, member);
    long cat_off = member * *args.num_catchments;
    long pix_off = member * *args.num_pixels;
    long link_off = member * *args.num_tributary_links;

    int p = args.tributary_pixel_idx_ptr[t];
    float old_dph = args.pixel_flood_depth_ptr[pix_off + p];
    float best_dph = old_dph;

    int start = args.tributary_edge_start_ptr[t];
    int end = (t + 1 < num_trib)
        ? args.tributary_edge_start_ptr[t + 1]
        : (int)(*args.num_tributary_links);

    for (int k = start; k < end; ++k) {
        int e = args.tributary_edge_order_ptr[k];
        int up_p = args.tributary_upstream_pixel_idx_ptr[e];
        int dn_p = args.tributary_downstream_pixel_idx_ptr[e];
        float dph1 = args.pixel_flood_depth_ptr[pix_off + up_p];
        float dph2 = args.pixel_flood_depth_ptr[pix_off + dn_p];
        float step = args.tributary_chain_step_ptr[(batched_tributary_chain_step ? link_off : 0) + e];
        float len = args.tributary_chain_length_ptr[(batched_tributary_chain_length ? link_off : 0) + e];
        float ddph = max((dph1 - dph2) / len, 0.0f);
        float cand = dph1 - ddph * step;
        if (cand > best_dph) {
            best_dph = cand;
        }
    }

    if (best_dph > old_dph) {
        args.pixel_flood_depth_ptr[pix_off + p] = best_dph;
        int ci = args.pixel_catchment_idx_ptr[p];
        atomic_fetch_max_explicit(
            args.catchment_tributary_modified_ptr + cat_off + ci,
            1,
            memory_order_relaxed);
    }

// HYDROFORGE METAL KERNEL BODY: compute_downscale_backwater_depth
    long num_bw = *args.num_backwater_pixels;
    long ensemble_size = *args.ensemble_size;
    long total = num_bw * ensemble_size;
    if ((long)i >= total) return;

    long b, member;
    cmf_decompose_index(i, num_bw, ensemble_size, b, member);
    long cat_off = member * *args.num_catchments;
    long pix_off = member * *args.num_pixels;
    long bw_off = member * num_bw;

    int p = args.backwater_pixel_idx_ptr[b];
    int ci = args.pixel_catchment_idx_ptr[p];
    if (args.catchment_tributary_modified_ptr[cat_off + ci] <= 0) return;

    int q = args.backwater_target_pixel_idx_ptr[b];
    long diff_off = batched_pixel_flood_diff ? pix_off : 0;
    float floor_elv = args.backwater_floor_elevation_ptr[(batched_backwater_floor_elevation ? bw_off : 0) + b];
    float dph1 = max(floor_elv, args.pixel_flood_diff_ptr[diff_off + q] + args.pixel_flood_depth_ptr[pix_off + q]);
    float flddif_p = args.pixel_flood_diff_ptr[diff_off + p];
    float cur_dph = args.pixel_flood_depth_ptr[pix_off + p];
    if (flddif_p + cur_dph < dph1) {
        args.pixel_flood_depth_ptr[pix_off + p] = dph1 - flddif_p;
    }

// HYDROFORGE METAL KERNEL BODY: compute_downscale_permanent_water
    long num_pixels = *args.num_pixels;
    long ensemble_size = *args.ensemble_size;
    long total = num_pixels * ensemble_size;
    if ((long)i >= total) return;

    long p, member;
    cmf_decompose_index(i, num_pixels, ensemble_size, p, member);
    (void)member;
    float perm_depth = *args.permanent_water_depth;
    long perm_idx = batched_pixel_is_permanent_water ? (long)i : p;
    if (perm_depth > 0.0f && args.pixel_is_permanent_water_ptr[perm_idx] != 0) {
        args.pixel_flood_depth_ptr[i] = max(perm_depth, args.pixel_flood_depth_ptr[i]);
    }

// HYDROFORGE METAL KERNEL BODY: compute_downscale_flood_duration
    long num_pixels = *args.num_pixels;
    long ensemble_size = *args.ensemble_size;
    long total = num_pixels * ensemble_size;
    if ((long)i >= total) return;

    long p, member;
    cmf_decompose_index(i, num_pixels, ensemble_size, p, member);
    long cat_off = member * *args.num_catchments;
    long pix_off = member * num_pixels;

    int ci = args.pixel_catchment_idx_ptr[p];
    float diffdph = args.pixel_flood_diff_ptr[(batched_pixel_flood_diff ? pix_off : 0) + p];
    float dur = 0.0f;

    if (diffdph >= 0.0f) {
        int bin_idx = (int)(diffdph / *args.duration_depth_interval);
        if (bin_idx >= num_duration_bins) {
            bin_idx = num_duration_bins - 1;
        }
        long table_off = batched_flood_duration_table ? (cat_off * (long)num_duration_bins) : 0;
        dur = args.flood_duration_table_ptr[table_off + (long)ci * (long)num_duration_bins + bin_idx];
    }

    float perm_dur = *args.permanent_water_duration;
    bool is_perm = args.pixel_is_permanent_water_ptr[(batched_pixel_is_permanent_water ? pix_off : 0) + p] != 0;
    if (perm_dur > 0.0f && is_perm) {
        dur = max(perm_dur, dur);
    }

    args.pixel_flood_duration_ptr[i] = dur;
