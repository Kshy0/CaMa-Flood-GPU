MODULE CMF_CALC_STONXT_MOD
!==========================================================
!* PURPOSE: calculate the storage in the next time step in FTCS diff. eq.
!
! (C) D.Yamazaki & E. Dutra  (U-Tokyo/FCUL)  Aug 2019
!
! Licensed under the Apache License, Version 2.0 (the "License");
!   You may not use this file except in compliance with the License.
!   You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software distributed under the License is 
!  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
! See the License for the specific language governing permissions and limitations under the License.
!
! Modified by Shengyu Kang to remove dependencies on other modules,
! making it easier to call from CaMa-Flood-GPU.
!==========================================================
  USE PARKIND1, ONLY: JPIM, JPRB, JPRD
  IMPLICIT NONE
  
  CONTAINS
!####################################################################
! -- CMF_CALC_STONXT
!
!####################################################################
  PURE SUBROUTINE CMF_CALC_STONXT(NSEQALL, DT, D2RIVOUT, D2FLDOUT, D2RUNOFF, &
                                  D2RIVINF, D2FLDINF, D2PTHOUT, D2FLDFRC, &
                                  D2RIVSTO, D2FLDSTO, D2OUTFLW, D2STORGE)
  IMPLICIT NONE
  
  ! Input parameters
  INTEGER(KIND=JPIM), INTENT(IN) :: NSEQALL
  REAL(KIND=JPRB), INTENT(IN) :: DT
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVOUT(NSEQALL,1), D2FLDOUT(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RUNOFF(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVINF(NSEQALL,1), D2FLDINF(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2PTHOUT(NSEQALL,1), D2FLDFRC(NSEQALL,1)
  
  ! Input/Output parameters
  REAL(KIND=JPRD), INTENT(INOUT) :: D2RIVSTO(NSEQALL,1), D2FLDSTO(NSEQALL,1)
  
  ! Output parameters
  REAL(KIND=JPRB), INTENT(OUT) :: D2OUTFLW(NSEQALL,1), D2STORGE(NSEQALL,1)

  
  ! Save for OpenMP
  INTEGER(KIND=JPIM) :: ISEQ
  REAL(KIND=JPRB) :: DRIVROF, DFLDROF
  !================================================

  !$OMP PARALLEL DO SIMD
  DO ISEQ=1, NSEQALL


    D2RIVSTO(ISEQ,1) = D2RIVSTO(ISEQ,1) + D2RIVINF(ISEQ,1)*DT - D2RIVOUT(ISEQ,1)*DT
    IF ( D2RIVSTO(ISEQ,1) < 0._JPRD ) THEN
      D2FLDSTO(ISEQ,1) = D2FLDSTO(ISEQ,1) + D2RIVSTO(ISEQ,1)
      D2RIVSTO(ISEQ,1) = 0._JPRD
    ENDIF

    D2FLDSTO(ISEQ,1) = D2FLDSTO(ISEQ,1) + D2FLDINF(ISEQ,1)*DT - D2FLDOUT(ISEQ,1)*DT &
                                        - D2PTHOUT(ISEQ,1)*DT
  !!                                      + D2PTHINF(ISEQ,1)*DT - D2PTHOUT(ISEQ,1)*DT  !! pthinf not used v4.3
    IF( D2FLDSTO(ISEQ,1) < 0._JPRD )THEN
      D2RIVSTO(ISEQ,1)=MAX( D2RIVSTO(ISEQ,1)+D2FLDSTO(ISEQ,1), 0._JPRD )
      D2FLDSTO(ISEQ,1)=0._JPRD
    ENDIF

    D2OUTFLW(ISEQ,1)=D2RIVOUT(ISEQ,1)+D2FLDOUT(ISEQ,1)
  !!  D2OUTFLW(ISEQ,1)=D2RIVOUT(ISEQ,1)+D2FLDOUT(ISEQ,1)+D2PTHOUT(ISEQ,1)   !! bug before v4.2 (pthout shoudl not be added)

    DRIVROF = ( D2RUNOFF(ISEQ,1) ) * ( 1._JPRB-D2FLDFRC(ISEQ,1) ) * DT
    DFLDROF = ( D2RUNOFF(ISEQ,1) ) *           D2FLDFRC(ISEQ,1)   * DT
    D2RIVSTO(ISEQ,1) = D2RIVSTO(ISEQ,1) + DRIVROF
    D2FLDSTO(ISEQ,1) = D2FLDSTO(ISEQ,1) + DFLDROF
    D2STORGE(ISEQ,1)= D2RIVSTO(ISEQ,1)+D2FLDSTO(ISEQ,1)

  END DO
  !$OMP END PARALLEL DO SIMD

  END SUBROUTINE CMF_CALC_STONXT

!####################################################################
END MODULE CMF_CALC_STONXT_MOD