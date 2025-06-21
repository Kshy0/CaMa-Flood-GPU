import numpy as np
import torch
from CMF_GPU.phys.fortran import cmf_calc_fldstg, cmf_calc_outflw, cmf_calc_pthout, cmf_calc_stonxt

CONVENTIONS = {
    "NSEQALL": lambda i: i["num_catchments"],
    "NLFP": lambda i: i["num_flood_levels"],
    "DFRCINC": lambda i: 1/i["num_flood_levels"],
    "D2GRAREA": lambda i: i["catchment_area"],
    "D2RIVLEN": lambda i: i["river_length"],
    "D2RIVWTH": lambda i: i["river_width"],
    "D2RIVELV": lambda i: i["catchment_elevation"] - i["river_height"],
    "D2RIVSTOMAX": lambda i: i["river_max_storage"],
    "D2FLDSTOMAX": lambda i: i["total_storage_table"][:, 1:],
    "D2FLDGRD": lambda i: i["flood_gradient_table"][:, :-1],
    "D2RIVSTO": lambda i: i["river_storage"],
    "D2RIVDPH": lambda i: i["river_depth"],
    "D2RIVDPH_PRE": lambda i: i["river_depth"],
    "D2FLDSTO": lambda i: i["flood_storage"],
    "D2FLDSTO_PRE": lambda i: i["flood_storage"],
    "D2FLDDPH": lambda i: i["flood_depth"],
    "D2STORGE": lambda i: i["outgoing_storage"],
    "NSEQRIV": lambda i: i["num_catchments"] - i["num_basins"],
    "PDSTMTH": lambda i: 10000.0,
    "PMANFLD": lambda i: 0.1,
    "PGRV": lambda i: i["gravity"],
    "I1NEXT": lambda i: i["downstream_idx"] + 1,
    "D2ELEVTN": lambda i: i["catchment_elevation"],
    "D2NXTDST": lambda i: i["downstream_distance"],
    "D2RIVHGT": lambda i: i["river_height"],
    "D2RIVMAN": lambda i: i["river_manning"],
    "D2DWNELV": lambda i: i["catchment_elevation"][i["downstream_idx"]],
    "D2RIVOUT": lambda i: i["river_outflow"],
    "D2RIVOUT_PRE": lambda i: i["river_outflow"],
    "D2FLDOUT_PRE": lambda i: i["flood_outflow"],
    "NPHOUT" : lambda i: i["num_bifurcation_paths"],
    "NPTHLEV" : lambda i: i["num_bifurcation_levels"],
    "PTH_UPST": lambda i: i["bifurcation_catchment_idx"] + 1,
    "PTH_DOWN": lambda i: i["bifurcation_downstream_idx"] + 1,
    "PTH_DST": lambda i: i["bifurcation_length"],
    "PTH_WTH": lambda i: i["bifurcation_width"],
    "PTH_ELV": lambda i: i["bifurcation_elevation"],
    "PTH_MAN": lambda i: i["bifurcation_manning"][0],
    "I2MASK": lambda i: torch.zeros_like(i["downstream_idx"]),
    "D1PTHFLW_PRE": lambda i: i["bifurcation_outflow"],
    "D1PTHFLW": lambda i: i["bifurcation_outflow"],
    "D1PTHFLWSUM": lambda i: i["bifurcation_outflow"].sum(axis=1),

}

def do_one_substep_fortran(
    input: dict,
    runoff,
    dT: float,
    enable_bifurcation: bool = True,
):
    (
        input["D2RIVOUT"],
        input["D2FLDOUT"],
        input["D2RIVVEL"],
        input["D2SFCELV_PRE"],
        input["D2SFCELV"],
    ) = cmf_calc_outflw.cmf_calc_outflw_mod.cmf_calc_outflw(
        nseqall=input["NSEQALL"],
        nseqriv=input["NSEQRIV"],
        dt=dT,
        pdstmth=input["PDSTMTH"],
        pmanfld=input["PMANFLD"],
        pgrv=input["PGRV"],
        i1next=input["I1NEXT"],
        d2rivelv=input["D2RIVELV"],
        d2elevtn=input["D2ELEVTN"],
        d2nxtdst=input["D2NXTDST"],
        d2rivwth=input["D2RIVWTH"],
        d2rivhgt=input["D2RIVHGT"],
        d2rivlen=input["D2RIVLEN"],
        d2rivman=input["D2RIVMAN"],
        d2rivdph=input["D2RIVDPH"],
        d2fldsto=input["D2FLDSTO"],
        d2storge=input["D2STORGE"],
        d2flddph=input["D2FLDDPH"],
        d2rivout_pre=input["D2RIVOUT_PRE"],
        d2rivdph_pre=input["D2RIVDPH_PRE"],
        d2fldout_pre=input["D2FLDOUT_PRE"],
        d2fldsto_pre=input["D2FLDSTO_PRE"],
    )


    if enable_bifurcation :
        (
            input["D1PTHFLW"],
            input["D1PTHFLWSUM"],
        ) = cmf_calc_pthout.cmf_calc_pthout_mod.cmf_calc_pthout(
            nseqall=input["NSEQALL"],
            npthout=input["NPHOUT"],
            npthlev=input["NPTHLEV"],
            dt=dT,
            pgrv=input["PGRV"],
            pth_upst=input["PTH_UPST"],
            pth_down=input["PTH_DOWN"],
            pth_dst=input["PTH_DST"],
            pth_elv=input["PTH_ELV"],
            pth_wth=input["PTH_WTH"],
            pth_man=input["PTH_MAN"],
            i2mask=input["I2MASK"],
            d1pthflw_pre=input["D1PTHFLW_PRE"],
            d2storge=input["D2STORGE"],
            d2sfcelv_pre=input["D2SFCELV_PRE"],
            d2sfcelv=input["D2SFCELV"],
        )

    (
        input["D2RIVINF"],
        input["D2FLDINF"],
        input["D2PTHOUT"],
    ) = cmf_calc_outflw.cmf_calc_outflw_mod.cmf_calc_inflow(
        nseqmax=input["NSEQALL"], 
        nseqall=input["NSEQALL"],
        nseqriv=input["NSEQRIV"],
        npthout=input["NPHOUT"],
        npthlev=input["NPTHLEV"],
        i2mask=input["I2MASK"],
        pth_upst=input["PTH_UPST"],
        pth_down=input["PTH_DOWN"],
        dt=dT,
        i1next=input["I1NEXT"],
        d2rivsto=input["D2RIVSTO"],
        d2fldsto=input["D2FLDSTO"],
        d2rivout=input["D2RIVOUT"],
        d2fldout=input["D2FLDOUT"],
        d1pthflw=input["D1PTHFLW"],
        d1pthflwsum=input["D1PTHFLWSUM"],
    )

    # save previous values for next step
    input["D2RIVDPH_PRE"] = input["D2RIVDPH"].copy()
    input["D2FLDSTO_PRE"] = input["D2FLDSTO"].copy()
    input["D2RIVOUT_PRE"] = input["D2RIVOUT"].copy()
    input["D2FLDOUT_PRE"] = input["D2FLDOUT"].copy()
    input["D1PTHFLW_PRE"] = input["D1PTHFLW"].copy()
    
    (
        input["D2OUTFLW"],
        input["D2STORGE"],
    ) = cmf_calc_stonxt.cmf_calc_stonxt_mod.cmf_calc_stonxt(
        nseqall=input["NSEQALL"],
        dt=dT,
        d2rivout=input["D2RIVOUT"], 
        d2fldout=input["D2FLDOUT"], 
        d2runoff=runoff,  
        d2rivinf=input["D2RIVINF"],
        d2fldinf=input["D2FLDINF"],
        d2pthout=input["D2PTHOUT"],
        d2fldfrc=input.get("D2FLDFRC", np.asfortranarray(np.zeros(input["NSEQALL"],dtype=np.float64))),
        d2rivsto=input["D2RIVSTO"], #
        d2fldsto=input["D2FLDSTO"], #
    )

    (
        input["D2RIVDPH"],
        input["D2FLDDPH"],
        input["D2FLDFRC"],
        input["D2FLDARE"],
        input["D2SFCELV"],
        input["D2STORGE"],
    ) = cmf_calc_fldstg.cmf_calc_fldstg_mod.cmf_calc_fldstg_def(
        nseqall=input["NSEQALL"],
        nlfp=input["NLFP"],
        dfrcinc=input["DFRCINC"],
        d2grarea=input["D2GRAREA"],
        d2rivlen=input["D2RIVLEN"],
        d2rivwth=input["D2RIVWTH"],
        d2rivelv=input["D2RIVELV"],
        d2rivstomax=input["D2RIVSTOMAX"],
        d2fldstomax=input["D2FLDSTOMAX"],
        d2fldgrd=input["D2FLDGRD"],
        d2rivsto=input["D2RIVSTO"], #
        d2fldsto=input["D2FLDSTO"], #
    )
