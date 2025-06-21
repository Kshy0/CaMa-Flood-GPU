MODULE CMF_CALC_PTHOUT_MOD
!==========================================================
!* PURPOSE: subroutine for bifurcation channel flow
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
! -- CMF_CALC_PTHOUT
! --
!####################################################################
  PURE SUBROUTINE CMF_CALC_PTHOUT(NSEQALL, NPTHOUT, NPTHLEV, DT, PGRV, &
                                  PTH_UPST, PTH_DOWN, PTH_DST, PTH_ELV, PTH_WTH, PTH_MAN, &
                                  I2MASK, D1PTHFLW_PRE, &
                                  D2STORGE, D2SFCELV_PRE, D2SFCELV, &
                                  D1PTHFLW, D1PTHFLWSUM)
  IMPLICIT NONE
  
  ! Input parameters
  INTEGER(KIND=JPIM), INTENT(IN) :: NSEQALL, NPTHOUT, NPTHLEV
  REAL(KIND=JPRB), INTENT(IN) :: DT, PGRV
  INTEGER(KIND=JPIM), INTENT(IN) :: PTH_UPST(NPTHOUT), PTH_DOWN(NPTHOUT)
  REAL(KIND=JPRB), INTENT(IN) :: PTH_DST(NPTHOUT)
  REAL(KIND=JPRB), INTENT(IN) :: PTH_ELV(NPTHOUT,NPTHLEV), PTH_WTH(NPTHOUT,NPTHLEV)
  REAL(KIND=JPRB), INTENT(IN) :: PTH_MAN(NPTHLEV)
  INTEGER(KIND=JPIM), INTENT(IN) :: I2MASK(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D1PTHFLW_PRE(NPTHOUT,NPTHLEV)
  REAL(KIND=JPRB), INTENT(IN) :: D2STORGE(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2SFCELV_PRE(NSEQALL,1)
  REAL(KIND=JPRB), INTENT(IN) :: D2SFCELV(NSEQALL,1)


  
  ! Output parameters
  REAL(KIND=JPRB), INTENT(OUT) :: D1PTHFLW(NPTHOUT,NPTHLEV)
  REAL(KIND=JPRB), INTENT(OUT) :: D1PTHFLWSUM(NPTHOUT)

  
  ! Save for OpenMP
  INTEGER(KIND=JPIM) :: IPTH, ILEV, ISEQP, JSEQP
  REAL(KIND=JPRB) :: DSLP, DFLW, DOUT_pr, DFLW_pr, DFLW_im, RATE
  !$OMP THREADPRIVATE        (DSLP, DFLW, DOUT_pr, DFLW_pr, DFLW_im, RATE, ILEV, ISEQP, JSEQP)
  !================================================

  D1PTHFLW(:,:) = 0._JPRB
  !$OMP PARALLEL DO
  DO IPTH=1, NPTHOUT  

    ISEQP=PTH_UPST(IPTH)
    JSEQP=PTH_DOWN(IPTH)
    !! Avoid calculation outside of domain
    IF (ISEQP<=0 .OR. JSEQP<=0 ) CYCLE
    IF (I2MASK(ISEQP,1)>0 .OR. I2MASK(JSEQP,1)>0 ) CYCLE  !! I2MASK is for 1: kinemacit 2: dam  no bifurcation
    
    DSLP  = (D2SFCELV(ISEQP,1)-D2SFCELV(JSEQP,1)) / PTH_DST(IPTH)
    DSLP = max(-0.005_JPRB,min(0.005_JPRB,DSLP))                                    !! v390 stabilization

    DO ILEV=1, NPTHLEV

      DFLW = MAX(D2SFCELV(ISEQP,1),D2SFCELV(JSEQP,1)) - PTH_ELV(IPTH,ILEV) 
      DFLW = MAX(DFLW,0._JPRB)

      DFLW_pr = MAX(D2SFCELV_PRE(ISEQP,1),D2SFCELV_PRE(JSEQP,1)) - PTH_ELV(IPTH,ILEV)
      DFLW_pr = MAX(DFLW_pr,0._JPRB)

      DFLW_im = (DFLW*DFLW_pr)**0.5_JPRB                                       !! semi implicit flow depth
      DFLW_im = MAX( DFLW_im,(DFLW*0.01_JPRB)**0.5_JPRB )

      IF( DFLW_im>1.E-5_JPRB )THEN                         !! local inertial equation, see [Bates et al., 2010, J.Hydrol.]
        DOUT_pr = D1PTHFLW_PRE(IPTH,ILEV) / PTH_WTH(IPTH,ILEV)              !! outflow (t-1) [m2/s] (unit width)
        D1PTHFLW(IPTH,ILEV) = PTH_WTH(IPTH,ILEV) * ( DOUT_pr + PGRV*DT*DFLW_im*DSLP ) &
               / ( 1._JPRB + PGRV*DT*PTH_MAN(ILEV)**2._JPRB * abs(DOUT_pr)*DFLW_im**(-7._JPRB/3._JPRB) )
      ELSE
        D1PTHFLW(IPTH,ILEV) = 0._JPRB
      ENDIF
    END DO

    D1PTHFLWSUM(IPTH)=0._JPRB
  END DO
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO SIMD
  DO IPTH=1, NPTHOUT
    D1PTHFLWSUM(IPTH) = SUM(D1PTHFLW(IPTH, :))
  END DO
  !$OMP END PARALLEL DO SIMD

  !! Storage change limitter (to prevent sudden increase of upstream water level) (v423)
  !$OMP PARALLEL DO
  DO IPTH=1, NPTHOUT  
    ISEQP=PTH_UPST(IPTH)
    JSEQP=PTH_DOWN(IPTH)
    IF( D1PTHFLWSUM(IPTH)/=0._JPRB )THEN
      RATE= 0.05_JPRB*min(D2STORGE(ISEQP,1),D2STORGE(JSEQP,1)) / abs(D1PTHFLWSUM(IPTH)*DT)  !! flow limit: 5% storage for stability
      RATE= min(RATE, 1.0_JPRB )
      D1PTHFLW(IPTH,:) =D1PTHFLW(IPTH,:) *RATE
      D1PTHFLWSUM(IPTH)=D1PTHFLWSUM(IPTH)*RATE
    ENDIF
  END DO
  !$OMP END PARALLEL DO

  END SUBROUTINE CMF_CALC_PTHOUT

!####################################################################
END MODULE CMF_CALC_PTHOUT_MOD