MODULE CMF_CALC_FLDSTG_MOD
!==========================================================
!* PURPOSE: calculate river and floodplain staging
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
! -- CMF_CALC_FLDSTG_DEF  !! default flood stage calculation
!####################################################################
  PURE SUBROUTINE CMF_CALC_FLDSTG_DEF(NSEQALL, NLFP, DFRCINC, &
                                      D2GRAREA, D2RIVLEN, D2RIVWTH, D2RIVELV, &
                                      D2RIVSTOMAX, D2FLDSTOMAX, D2FLDGRD, &
                                      D2RIVSTO, D2FLDSTO, &
                                      D2RIVDPH, D2FLDDPH, D2FLDFRC, D2FLDARE, &
                                      D2SFCELV, D2STORGE)
  IMPLICIT NONE
  
  ! Input parameters
  INTEGER(KIND=JPIM), INTENT(IN) :: NSEQALL, NLFP
  REAL(KIND=JPRB), INTENT(IN) :: DFRCINC
  REAL(KIND=JPRB), INTENT(IN) :: D2GRAREA(NSEQALL,1), D2RIVLEN(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVWTH(NSEQALL,1), D2RIVELV(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVSTOMAX(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2FLDSTOMAX(NSEQALL,NLFP)
  REAL(KIND=JPRB), INTENT(IN) :: D2FLDGRD(NSEQALL,NLFP)
  REAL(KIND=JPRD), INTENT(INOUT) :: D2RIVSTO(NSEQALL,1), D2FLDSTO(NSEQALL,1)
  
  ! Output parameters
  REAL(KIND=JPRB), INTENT(OUT) :: D2RIVDPH(NSEQALL,1), D2FLDDPH(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(OUT) :: D2FLDFRC(NSEQALL,1), D2FLDARE(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(OUT) :: D2SFCELV(NSEQALL,1), D2STORGE(NSEQALL,1)
  
  !*** LOCAL
  INTEGER(KIND=JPIM) :: ISEQ, I
  REAL(KIND=JPRD) :: DSTOALL
  REAL(KIND=JPRB) :: DSTO_add, DSTO_fil, DWTH_add, DWTH_fil
  REAL(KIND=JPRB) :: DDPH_fil, DWTH_inc
  !$OMP THREADPRIVATE        (I,DSTOALL,DSTO_add,DSTO_fil,DWTH_add,DWTH_fil,DDPH_fil,DWTH_inc,DSTOALL)
  !================================================
  
  
  ! Estimate water depth and flood extent from water storage
  !   Solution for Equations (1) and (2) in [Yamazaki et al. 2011 WRR].
  
  !$OMP PARALLEL DO
  DO ISEQ=1, NSEQALL
    !
    DSTOALL = D2RIVSTO(ISEQ,1) + D2FLDSTO(ISEQ,1)

    IF(DSTOALL > D2RIVSTOMAX(ISEQ,1))THEN
      I = 1
      DSTO_fil = D2RIVSTOMAX(ISEQ,1)
      DWTH_fil = D2RIVWTH(ISEQ,1)
      DDPH_fil = 0._JPRB
      DWTH_inc = D2GRAREA(ISEQ,1) / D2RIVLEN(ISEQ,1) * DFRCINC
      DO WHILE(DSTOALL > D2FLDSTOMAX(ISEQ,I) .AND. I <= NLFP)
        DSTO_fil = D2FLDSTOMAX(ISEQ,I)
        DWTH_fil = DWTH_fil + DWTH_inc
        DDPH_fil = DDPH_fil + D2FLDGRD(ISEQ,I) * DWTH_inc
        I = I + 1
        IF(I > NLFP) EXIT
      END DO
      IF(I > NLFP)THEN
        DSTO_add = DSTOALL - DSTO_fil
        DWTH_add = 0._JPRB
        D2FLDDPH(ISEQ,1) = DDPH_fil + DSTO_add / DWTH_fil / D2RIVLEN(ISEQ,1)
      ELSE
        DSTO_add = DSTOALL - DSTO_fil
        DWTH_add = -DWTH_fil + &
&        (DWTH_fil**2._JPRB + 2._JPRB * DSTO_add / D2RIVLEN(ISEQ,1) / D2FLDGRD(ISEQ,I))**0.5_JPRB
        D2FLDDPH(ISEQ,1) = DDPH_fil + D2FLDGRD(ISEQ,I) * DWTH_add
      ENDIF
      D2RIVSTO(ISEQ,1) = D2RIVSTOMAX(ISEQ,1) + D2RIVLEN(ISEQ,1) * D2RIVWTH(ISEQ,1) * D2FLDDPH(ISEQ,1)
      D2RIVSTO(ISEQ,1) = MIN(D2RIVSTO(ISEQ,1), DSTOALL)

      D2RIVDPH(ISEQ,1) = D2RIVSTO(ISEQ,1) / D2RIVLEN(ISEQ,1) / D2RIVWTH(ISEQ,1)
      !
      D2FLDSTO(ISEQ,1) = DSTOALL - D2RIVSTO(ISEQ,1)
      D2FLDSTO(ISEQ,1) = MAX(D2FLDSTO(ISEQ,1), 0._JPRD)
      D2FLDFRC(ISEQ,1) = (-D2RIVWTH(ISEQ,1) + DWTH_fil + DWTH_add) / (DWTH_inc * NLFP)
      D2FLDFRC(ISEQ,1) = MAX(D2FLDFRC(ISEQ,1), 0._JPRB)
      D2FLDFRC(ISEQ,1) = MIN(D2FLDFRC(ISEQ,1), 1._JPRB)
      D2FLDARE(ISEQ,1) = D2GRAREA(ISEQ,1) * D2FLDFRC(ISEQ,1)
    ELSE
      D2RIVSTO(ISEQ,1) = DSTOALL
      D2RIVDPH(ISEQ,1) = DSTOALL / D2RIVLEN(ISEQ,1) / D2RIVWTH(ISEQ,1)
      D2RIVDPH(ISEQ,1) = MAX(D2RIVDPH(ISEQ,1), 0._JPRB)
      D2FLDSTO(ISEQ,1) = 0._JPRD
      D2FLDDPH(ISEQ,1) = 0._JPRB
      D2FLDFRC(ISEQ,1) = 0._JPRB
      D2FLDARE(ISEQ,1) = 0._JPRB
    ENDIF
  END DO
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO SIMD
  DO ISEQ=1, NSEQALL
    D2SFCELV(ISEQ,1) = D2RIVELV(ISEQ,1) + D2RIVDPH(ISEQ,1)
    D2STORGE(ISEQ,1) = D2RIVSTO(ISEQ,1) + D2FLDSTO(ISEQ,1)
  END DO
  !$OMP END PARALLEL DO SIMD
  
  END SUBROUTINE CMF_CALC_FLDSTG_DEF
  
END MODULE CMF_CALC_FLDSTG_MOD