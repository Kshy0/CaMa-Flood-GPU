MODULE CMF_CALC_OUTFLW_MOD
!==========================================================
!* PURPOSE: CaMa-Flood physics for river&floodplain discharge
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
! -- CMF_CALC_OUTFLW
! -- CMF_CALC_INFLOW
! --
!####################################################################
  PURE SUBROUTINE CMF_CALC_OUTFLW(NSEQALL, NSEQRIV, DT, PDSTMTH, PMANFLD, PGRV, &
                                  I1NEXT, D2RIVELV, D2ELEVTN, D2NXTDST, D2RIVWTH, &
                                  D2RIVHGT, D2RIVLEN, D2RIVMAN,  &
                                  D2RIVDPH, D2FLDSTO, D2STORGE, D2FLDDPH, &
                                  D2RIVOUT_PRE, D2RIVDPH_PRE, D2FLDOUT_PRE, D2FLDSTO_PRE, &
                                  D2RIVOUT, D2FLDOUT, D2RIVVEL, D2SFCELV_PRE, D2SFCELV)
  IMPLICIT NONE
  
  ! Input parameters
  INTEGER(KIND=JPIM), INTENT(IN) :: NSEQALL, NSEQRIV
  REAL(KIND=JPRB), INTENT(IN) :: DT, PDSTMTH, PMANFLD, PGRV
  INTEGER(KIND=JPIM), INTENT(IN) :: I1NEXT(NSEQALL)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVELV(NSEQALL,1), D2ELEVTN(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2NXTDST(NSEQALL,1), D2RIVWTH(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVHGT(NSEQALL,1), D2RIVLEN(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVMAN(NSEQALL,1), D2RIVDPH(NSEQALL,1)
  REAL(KIND=JPRD), INTENT(IN) :: D2FLDSTO(NSEQALL,1), D2STORGE(NSEQALL,1), D2FLDDPH(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2RIVOUT_PRE(NSEQALL,1), D2RIVDPH_PRE(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2FLDOUT_PRE(NSEQALL,1), D2FLDSTO_PRE(NSEQALL,1)
  
  ! Output parameters
  REAL(KIND=JPRB), INTENT(OUT) :: D2RIVOUT(NSEQALL,1), D2FLDOUT(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(OUT) :: D2RIVVEL(NSEQALL,1), D2SFCELV_PRE(NSEQALL,1) ,D2SFCELV(NSEQALL,1)

  ! Local variables
  REAL(KIND=JPRB) :: D2DWNELV_PRE(NSEQALL,1), D2DWNELV(NSEQALL,1)
  REAL(KIND=JPRB) :: D2FLDDPH_PRE(NSEQALL,1)
  
  ! save for OpenMP
  INTEGER(KIND=JPIM) :: ISEQ, JSEQ
  REAL(KIND=JPRB) :: DFSTO,DSFC,DSFC_pr,DSLP,DFLW,DFLW_pr,DFLW_im,DARE,DARE_pr,DARE_im,DOUT_pr,DOUT,DVEL
  LOGICAL :: Mask
  REAL(KIND=JPRB) :: RATE
  !$OMP THREADPRIVATE          (JSEQ)
  !================================================

  ! calculate water surface elevation
  !$OMP PARALLEL DO SIMD
  DO ISEQ=1, NSEQALL
    D2SFCELV(ISEQ,1)     = D2RIVELV(ISEQ,1) + D2RIVDPH(ISEQ,1)
    D2SFCELV_PRE(ISEQ,1) = D2RIVELV(ISEQ,1) + D2RIVDPH_PRE(ISEQ,1)
    D2FLDDPH_PRE(ISEQ,1) = MAX( D2RIVDPH_PRE(ISEQ,1)-D2RIVHGT(ISEQ,1), 0._JPRB )
  END DO
  !$OMP END PARALLEL DO SIMD

  !Update downstream elevation
  !$OMP PARALLEL DO
  DO ISEQ=1, NSEQRIV
    JSEQ=I1NEXT(ISEQ) ! next cell's pixel
    D2DWNELV(ISEQ,1)     = D2SFCELV(JSEQ,1)
    D2DWNELV_PRE(ISEQ,1) = D2SFCELV_PRE(JSEQ,1)
  END DO
  !$OMP END PARALLEL DO
  
  !$OMP PARALLEL DO
  DO ISEQ=NSEQRIV+1, NSEQALL
    D2DWNELV(ISEQ,1)     = D2ELEVTN(ISEQ,1)
    D2DWNELV_PRE(ISEQ,1) = D2ELEVTN(ISEQ,1)
  END DO
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO SIMD PRIVATE(DSFC,DSFC_pr,DSLP,DFLW,DFLW_pr,DFLW_im,DARE,DARE_pr,DARE_im,DOUT_pr,DOUT,DVEL,Mask)
  DO ISEQ=1, NSEQRIV                                                !! for normal cells
    DSFC = MAX( D2SFCELV(ISEQ,1),    D2DWNELV(ISEQ,1) )
    DSLP = ( D2SFCELV(ISEQ,1)-D2DWNELV(ISEQ,1) ) * D2NXTDST(ISEQ,1)**(-1._JPRB)
  !=== River Flow ===
    DFLW = DSFC - D2RIVELV(ISEQ,1)                             !!  flow cross-section depth
    DARE = MAX( D2RIVWTH(ISEQ,1)*DFLW, 1.E-10_JPRB )           !!  flow cross-section area

    DSFC_pr=MAX( D2SFCELV_PRE(ISEQ,1),D2DWNELV_PRE(ISEQ,1) )
    DFLW_pr=DSFC_pr - D2RIVELV(ISEQ,1)
    DFLW_im=MAX( (DFLW*DFLW_pr)**0.5_JPRB ,1.E-6_JPRB )             !! semi implicit flow depth

    DOUT_pr= D2RIVOUT_PRE(ISEQ,1) * D2RIVWTH(ISEQ,1)**(-1._JPRB)    !! outflow (t-1) [m2/s] (unit width)
    DOUT=D2RIVWTH(ISEQ,1) * ( DOUT_pr + PGRV*DT*DFLW_im*DSLP ) &
          * ( 1._JPRB + PGRV*DT*D2RIVMAN(ISEQ,1)**2._JPRB*abs(DOUT_pr)*DFLW_im**(-7._JPRB/3._JPRB) )**(-1._JPRB)
    DVEL= D2RIVOUT(ISEQ,1) * DARE**(-1._JPRB)

    Mask=(DFLW_im>1.E-5 .and. DARE>1.E-5)   !! replace small depth location with zero
    D2RIVOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)
    D2RIVVEL(ISEQ,1) = merge( DVEL, 0._JPRB, Mask)
  END DO
  !$OMP END PARALLEL DO SIMD

  !=== Floodplain Flow ===

  !$OMP PARALLEL DO SIMD PRIVATE(DFSTO,DSFC,DSFC_pr,DSLP,DFLW,DFLW_pr,DFLW_im,DARE,DARE_pr,DARE_im,DOUT_pr,DOUT,Mask)
  DO ISEQ=1, NSEQRIV      
    DFSTO=  D2FLDSTO(ISEQ,1)
    DSFC   = MAX( D2SFCELV(ISEQ,1),    D2DWNELV(ISEQ,1) )
    DSLP   = ( D2SFCELV(ISEQ,1)-D2DWNELV(ISEQ,1) ) * D2NXTDST(ISEQ,1)**(-1._JPRB)
    DSLP   = MAX( -0.005_JPRB, min( 0.005_JPRB, DSLP ))    !! set max&min [instead of using weir equation for efficiency]

    DFLW   = MAX( DSFC-D2ELEVTN(ISEQ,1), 0._JPRB )
    DARE   = DFSTO * D2RIVLEN(ISEQ,1)**(-1._JPRB)
    DARE   = MAX( DARE - D2FLDDPH(ISEQ,1)*D2RIVWTH(ISEQ,1), 0._JPRB )   !! remove above river channel area
  
    DSFC_pr = MAX( D2SFCELV_PRE(ISEQ,1),D2DWNELV_PRE(ISEQ,1) )
    DFLW_pr = DSFC_pr - D2ELEVTN(ISEQ,1)
    DFLW_im = MAX( (MAX(DFLW*DFLW_pr,0._JPRB))**0.5_JPRB, 1.E-6_JPRB )
  
    DARE_pr = D2FLDSTO_PRE(ISEQ,1) * D2RIVLEN(ISEQ,1)**(-1._JPRB)
    DARE_pr = MAX( DARE_pr - D2FLDDPH_PRE(ISEQ,1)*D2RIVWTH(ISEQ,1), 1.E-6_JPRB )   !! remove above river channel area
    DARE_im = MAX( (DARE*DARE_pr)**0.5_JPRB, 1.E-6_JPRB )

    DOUT_pr = D2FLDOUT_PRE(ISEQ,1)
    DOUT = (DOUT_pr + PGRV*DT*DARE_im*DSLP) &
          * (1._JPRB +PGRV*DT*PMANFLD**2._JPRB * abs(DOUT_pr)*DFLW_im**(-4._JPRB/3._JPRB)*DARE_im**(-1._JPRB) )**(-1._JPRB)

    Mask=(DFLW_im>1.E-5_JPRB .and. DARE>1.E-5_JPRB)  !! replace small depth location with zero
    D2FLDOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)

    DOUT=D2FLDOUT(ISEQ,1)
    Mask=( D2FLDOUT(ISEQ,1)*D2RIVOUT(ISEQ,1)>0._JPRB ) !! river and floodplain different direction
    D2FLDOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)
  END DO
  !$OMP END PARALLEL DO SIMD

  !=== river mouth flow ===
  !$OMP PARALLEL DO SIMD PRIVATE(DSFC,DSFC_pr,DSLP,DFLW,DFLW_pr,DFLW_im,DARE,DARE_pr,DARE_im,DOUT_pr,DOUT,DVEL,Mask)
  DO ISEQ=NSEQRIV+1, NSEQALL
    DSLP = ( D2SFCELV(ISEQ,1) - D2DWNELV(ISEQ,1) ) * PDSTMTH ** (-1._JPRB)

    DFLW = D2RIVDPH(ISEQ,1)
    DARE = D2RIVWTH(ISEQ,1) * DFLW
    DARE = MAX( DARE, 1.E-10_JPRB )             !!  flow cross-section area (min value for stability)

    DFLW_pr=D2RIVDPH_PRE(ISEQ,1)
    DFLW_im=MAX( (DFLW*DFLW_pr)**0.5_JPRB, 1.E-6_JPRB )                                    !! semi implicit flow depth

    DOUT_pr = D2RIVOUT_PRE(ISEQ,1) * D2RIVWTH(ISEQ,1)**(-1._JPRB)
    DOUT = D2RIVWTH(ISEQ,1) * ( DOUT_pr + PGRV*DT*DFLW_im*DSLP ) &
             * ( 1._JPRB + PGRV*DT*D2RIVMAN(ISEQ,1)**2._JPRB * abs(DOUT_pr)*DFLW_im**(-7._JPRB/3._JPRB) )**(-1._JPRB)
    DVEL = D2RIVOUT(ISEQ,1) * DARE**(-1._JPRB)

    Mask=(DFLW_im>1.E-5 .and. DARE>1.E-5)   !! replace small depth location with zero
    D2RIVOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)
    D2RIVVEL(ISEQ,1) = merge( DVEL, 0._JPRB, Mask)
  END DO
  !$OMP END PARALLEL DO SIMD

  !=== floodplain mouth flow ===

  !$OMP PARALLEL DO SIMD PRIVATE(DFSTO,DSFC,DSFC_pr,DSLP,DFLW,DFLW_pr,DFLW_im,DARE,DARE_pr,DARE_im,DOUT_pr,DOUT,Mask)
  DO ISEQ=NSEQRIV+1, NSEQALL
    DFSTO = D2FLDSTO(ISEQ,1)
    DSLP = ( D2SFCELV(ISEQ,1) - D2DWNELV(ISEQ,1) ) * PDSTMTH ** (-1._JPRB)
    DSLP = max( -0.005_JPRB, min( 0.005_JPRB,DSLP ))    !! set max&min [instead of using weir equation for efficiency]

    DFLW = D2SFCELV(ISEQ,1)-D2ELEVTN(ISEQ,1)
    DARE = MAX( DFSTO * D2RIVLEN(ISEQ,1)**(-1._JPRB) - D2FLDDPH(ISEQ,1)*D2RIVWTH(ISEQ,1), 0._JPRB ) !! remove above channel
  
    DFLW_pr = D2SFCELV_PRE(ISEQ,1)-D2ELEVTN(ISEQ,1)
    DFLW_im = MAX( (MAX(DFLW*DFLW_pr,0._JPRB))**0.5_JPRB, 1.E-6_JPRB )
  
    DARE_pr = MAX( D2FLDSTO_PRE(ISEQ,1) * D2RIVLEN(ISEQ,1)**(-1._JPRB) - D2FLDDPH_PRE(ISEQ,1)*D2RIVWTH(ISEQ,1), 1.E-6_JPRB )   
    DARE_im = MAX( (DARE*DARE_pr)**0.5_JPRB, 1.E-6_JPRB )
  
    DOUT_pr = D2FLDOUT_PRE(ISEQ,1)
    DOUT = ( DOUT_pr + PGRV*DT*DARE_im*DSLP ) &
            * (1._JPRB + PGRV*DT*PMANFLD**2._JPRB * abs(DOUT_pr)*DFLW_im**(-4._JPRB/3._JPRB)*DARE_im**(-1._JPRB) )**(-1._JPRB)

    Mask=(DFLW_im>1.E-5 .and. DARE>1.E-5)   !! replace small depth location with zero
    D2FLDOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)

    DOUT=D2FLDOUT(ISEQ,1)
    Mask=( D2FLDOUT(ISEQ,1)*D2RIVOUT(ISEQ,1)>0._JPRB ) !! river and floodplain different direction
    D2FLDOUT(ISEQ,1) = merge( DOUT, 0._JPRB, Mask)
  END DO
  !$OMP END PARALLEL DO SIMD

  !$OMP PARALLEL DO SIMD PRIVATE(RATE,DOUT)
  DO ISEQ=1, NSEQRIV
    !! Storage change limitter to prevent sudden increase of "upstream" water level when backwardd flow (v423)
    DOUT = max( (-D2RIVOUT(ISEQ,1)-D2FLDOUT(ISEQ,1))*DT, 1.E-10 )
    RATE = min( 0.05_JPRB*D2STORGE(ISEQ,1)/DOUT, 1._JPRB)
    D2RIVOUT(ISEQ,1)=D2RIVOUT(ISEQ,1)*RATE
    D2FLDOUT(ISEQ,1)=D2FLDOUT(ISEQ,1)*RATE
  END DO
  !$OMP END PARALLEL DO SIMD

  END SUBROUTINE CMF_CALC_OUTFLW

!####################################################################
!+
!+
!+
!####################################################################
  PURE SUBROUTINE CMF_CALC_INFLOW(NSEQMAX, NSEQALL, NSEQRIV, NPTHOUT, NPTHLEV, I2MASK, PTH_UPST, PTH_DOWN, DT, &
                                  I1NEXT, D2RIVSTO, D2FLDSTO, D2RIVOUT, D2FLDOUT, D1PTHFLW, D1PTHFLWSUM,&
                                  D2RIVINF, D2FLDINF, D2PTHOUT)
  IMPLICIT NONE
  
  ! Input parameters
  INTEGER(KIND=JPIM), INTENT(IN) :: NSEQMAX, NSEQALL, NSEQRIV, NPTHOUT, NPTHLEV
  INTEGER(KIND=JPIM), INTENT(IN) :: I2MASK(NSEQMAX,1), PTH_UPST(NPTHOUT), PTH_DOWN(NPTHOUT)
  REAL(KIND=JPRB), INTENT(IN) :: DT
  INTEGER(KIND=JPIM), INTENT(IN) :: I1NEXT(NSEQALL)
  REAL(KIND=JPRD), INTENT(IN) :: D2RIVSTO(NSEQALL,1), D2FLDSTO(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(INOUT) :: D2RIVOUT(NSEQALL,1), D2FLDOUT(NSEQALL,1), D1PTHFLW(NPTHOUT, NPTHLEV), D1PTHFLWSUM(NPTHOUT)

  ! Output parameters
  REAL(KIND=JPRB), INTENT(OUT) :: D2RIVINF(NSEQALL,1), D2FLDINF(NSEQALL,1), D2PTHOUT(NSEQALL,1)

  !*** Local
  REAL(KIND=JPRD) :: D2STOOUT(NSEQMAX,1)                      !! total outflow from a grid     [m3]
  REAL(KIND=JPRB) :: D2RATE(NSEQMAX,1)                        !! outflow correction
  
  ! SAVE for OpenMP
  INTEGER(KIND=JPIM) :: ISEQ, JSEQ, IPTH, ILEV, ISEQP, JSEQP
  REAL(KIND=JPRB) :: OUT_R1, OUT_R2, OUT_F1, OUT_F2, DIUP, DIDW
  !$OMP THREADPRIVATE     (JSEQ,OUT_R1, OUT_R2, OUT_F1, OUT_F2, DIUP, DIDW)
  !================================================
    
  !*** 1. initialize & calculate D2STOOUT for normal cells

  !$OMP PARALLEL DO
  DO ISEQ=1, NSEQALL
    D2RIVINF(ISEQ,1) = 0._JPRD
    D2FLDINF(ISEQ,1) = 0._JPRD
    D2PTHOUT(ISEQ,1) = 0._JPRD
    D2STOOUT(ISEQ,1) = 0._JPRD
    D2RATE(ISEQ,1) = 1._JPRB
  END DO
  !$OMP END PARALLEL DO

  !! for normal cells ---------
  !$OMP PARALLEL DO
  DO ISEQ=1, NSEQRIV                                                    !! for normalcells
    JSEQ=I1NEXT(ISEQ) ! next cell's pixel
    OUT_R1 = max(  D2RIVOUT(ISEQ,1),0._JPRB )
    OUT_R2 = max( -D2RIVOUT(ISEQ,1),0._JPRB )
    OUT_F1 = max(  D2FLDOUT(ISEQ,1),0._JPRB )
    OUT_F2 = max( -D2FLDOUT(ISEQ,1),0._JPRB )
    DIUP=(OUT_R1+OUT_F1)*DT
    DIDW=(OUT_R2+OUT_F2)*DT
  !$OMP ATOMIC
    D2STOOUT(ISEQ,1) = D2STOOUT(ISEQ,1) + DIUP 
  !$OMP ATOMIC
    D2STOOUT(JSEQ,1) = D2STOOUT(JSEQ,1) + DIDW 
  END DO
  !$OMP END PARALLEL DO

  !! for river mouth grids ------------
  !$OMP PARALLEL DO
  DO ISEQ=NSEQRIV+1, NSEQALL
    OUT_R1 = max( D2RIVOUT(ISEQ,1), 0._JPRB )
    OUT_F1 = max( D2FLDOUT(ISEQ,1), 0._JPRB )
    D2STOOUT(ISEQ,1) = D2STOOUT(ISEQ,1) + OUT_R1*DT + OUT_F1*DT
  END DO
  !$OMP END PARALLEL DO

  !! for bifurcation channels ------------
  !$OMP PARALLEL DO  !! No OMP Atomic for bit-identical simulation (set in Mkinclude)
    DO IPTH=1, NPTHOUT  
      ISEQP=PTH_UPST(IPTH)
      JSEQP=PTH_DOWN(IPTH)
      !! Avoid calculation outside of domain
      IF (ISEQP<=0 .OR. JSEQP<=0 ) CYCLE
      IF (I2MASK(ISEQP,1)>0 .OR. I2MASK(JSEQP,1)>0 ) CYCLE  !! I2MASK is for 1: kinemacit 2: dam  no bifurcation
    
      OUT_R1 = max(  D1PTHFLWSUM(IPTH),0._JPRB )
      OUT_R2 = max( -D1PTHFLWSUM(IPTH),0._JPRB )
      DIUP=(OUT_R1)*DT
      DIDW=(OUT_R2)*DT
  !$OMP ATOMIC
      D2STOOUT(ISEQP,1) = D2STOOUT(ISEQP,1) + DIUP
  !$OMP ATOMIC
      D2STOOUT(JSEQP,1) = D2STOOUT(JSEQP,1) + DIDW
    END DO
  !$OMP END PARALLEL DO  !! No OMP Atomic for bit-identical simulation (set in Mkinclude)

  !============================
  !*** 2. modify outflow

  !$OMP PARALLEL DO
  DO ISEQ=1, NSEQALL
    IF ( D2STOOUT(ISEQ,1) > 1.E-8 ) THEN
      D2RATE(ISEQ,1) =  min( (D2RIVSTO(ISEQ,1)+D2FLDSTO(ISEQ,1)) / D2STOOUT(ISEQ,1),1._JPRD )
    ENDIF
  END DO
  !$OMP END PARALLEL DO

  !! normal pixels------
  !$OMP PARALLEL DO  !! No OMP Atomic for bit-identical simulation (set in Mkinclude)
  DO ISEQ=1, NSEQRIV ! for normal pixels
    JSEQ=I1NEXT(ISEQ)
    IF( D2RIVOUT(ISEQ,1) >= 0._JPRB )THEN
      D2RIVOUT(ISEQ,1) = D2RIVOUT(ISEQ,1)*D2RATE(ISEQ,1)
      D2FLDOUT(ISEQ,1) = D2FLDOUT(ISEQ,1)*D2RATE(ISEQ,1)
    ELSE
      D2RIVOUT(ISEQ,1) = D2RIVOUT(ISEQ,1)*D2RATE(JSEQ,1)
      D2FLDOUT(ISEQ,1) = D2FLDOUT(ISEQ,1)*D2RATE(JSEQ,1)
    ENDIF
  !$OMP ATOMIC
    D2RIVINF(JSEQ,1) = D2RIVINF(JSEQ,1) + D2RIVOUT(ISEQ,1)             !! total inflow to a grid (from upstream)
  !$OMP ATOMIC
    D2FLDINF(JSEQ,1) = D2FLDINF(JSEQ,1) + D2FLDOUT(ISEQ,1)
  END DO
  !$OMP END PARALLEL DO

  !! river mouth-----------------
  !$OMP PARALLEL DO
  DO ISEQ=NSEQRIV+1, NSEQALL
    D2RIVOUT(ISEQ,1) = D2RIVOUT(ISEQ,1)*D2RATE(ISEQ,1)
    D2FLDOUT(ISEQ,1) = D2FLDOUT(ISEQ,1)*D2RATE(ISEQ,1)
  END DO
  !$OMP END PARALLEL DO

  !! bifurcation channels --------

  !$OMP PARALLEL DO  !! No OMP Atomic for bit-identical simulation (set in Mkinclude)
    DO IPTH=1, NPTHOUT  
      ISEQP=PTH_UPST(IPTH)
      JSEQP=PTH_DOWN(IPTH)
      !! Avoid calculation outside of domain
      IF (ISEQP<=0 .OR. JSEQP<=0 ) CYCLE
      IF (I2MASK(ISEQP,1)>0 .OR. I2MASK(JSEQP,1)>0 ) CYCLE  !! I2MASK is for 1: kinemacit 2: dam  no bifurcation
    
      DO ILEV=1, NPTHLEV
        IF( D1PTHFLW(IPTH,ILEV) >= 0._JPRB )THEN                                  !! total outflow from each grid
          D1PTHFLW(IPTH,ILEV) = D1PTHFLW(IPTH,ILEV)*D2RATE(ISEQP,1)
        ELSE
          D1PTHFLW(IPTH,ILEV) = D1PTHFLW(IPTH,ILEV)*D2RATE(JSEQP,1)
        ENDIF
      END DO

      IF( D1PTHFLWSUM(IPTH) >= 0._JPRB )THEN                                  !! total outflow from each grid
        D1PTHFLWSUM(IPTH) = D1PTHFLWSUM(IPTH)*D2RATE(ISEQP,1)
      ELSE
        D1PTHFLWSUM(IPTH) = D1PTHFLWSUM(IPTH)*D2RATE(JSEQP,1)
      ENDIF    

  !$OMP ATOMIC
      D2PTHOUT(ISEQP,1) = D2PTHOUT(ISEQP,1) + D1PTHFLWSUM(IPTH)
  !$OMP ATOMIC
      D2PTHOUT(JSEQP,1) = D2PTHOUT(JSEQP,1) - D1PTHFLWSUM(IPTH)
    END DO
  !$OMP END PARALLEL DO  !! No OMP Atomic for bit-identical simulation (set in Mkinclude)


  END SUBROUTINE CMF_CALC_INFLOW

END MODULE CMF_CALC_OUTFLW_MOD