#!/bin/bash

unset DISPLAY
echo Running on host `hostname`
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
path="/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v07/v0701/systematics-full/nominal/"
cd /afs/cern.ch/user/v/vandergr/private/DNN/source; source setup.sh;
ttree2hdf5 "$path"mc16a/410470.root "$path"mc16d/410470.root "$path"mc16e/410470.root --branch-regex "nJets_OR|jet_pseudoscore_DL1r0|jet_pseudoscore_DL1r1|jet_pseudoscore_DL1r2|bjet_pt0|DRll01|lep_Pt_0|lep_Pt_1|sumPsbtag|HT_jets|jet_pt0_nofwd|met_met|MtLepMet|HT_lep|DEtall_SS|DPhill_SS|eventNumber|totalEventsWeighted|RunYear|weight_pileup|jvtSF_customOR|bTagSF_weight_DL1r_77|weight_mc|xs|Mll01|nJets_OR_DL1r_77|nJets_OR_DL1r_70|nJets_OR_DL1r_60|nJets_OR_DL1r_85" -s "(dilep_type && custTrigMatch_TightElMediumMuID_FCLooseIso_SLTorDLT && ((((abs(lep_ID_0) == 13 && lep_isMedium_0 && lep_isolationFCLoose_0 && passPLIVTight_0) || (abs(lep_ID_0) == 11 && lep_isolationFCLoose_0 && lep_isTightLH_0 && lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 && lep_ambiguityType_0 == 0 && passPLIVTight_0)) && ((abs(lep_ID_1) == 13 && lep_isMedium_1 && lep_isolationFCLoose_1 && passPLIVTight_1) || (abs(lep_ID_1) == 11 && lep_isolationFCLoose_1 && lep_isTightLH_1 && lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 && lep_ambiguityType_1 == 0 && passPLIVTight_1)) && nJets_OR_DL1r_77>=2 ) || (((abs(lep_ID_0) == 13 && lep_isMedium_0 && lep_isolationFCLoose_0 && passPLIVVeryTight_0) || (abs(lep_ID_0) == 11 && lep_isolationFCLoose_0 && lep_isTightLH_0 && lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 && lep_ambiguityType_0 == 0 && passPLIVVeryTight_0)) && ((abs(lep_ID_1) == 13 && lep_isMedium_1 && lep_isolationFCLoose_1 && passPLIVVeryTight_1) || (abs(lep_ID_1) == 11 && lep_isolationFCLoose_1 && lep_isTightLH_1 && lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 && lep_ambiguityType_1 == 0 && passPLIVVeryTight_1)) && nJets_OR_DL1r_77==1 )) && lep_Pt_0*1e-3>20 && lep_Pt_1*1e-3>20 && nJets_OR>=2 && lep_ID_0*lep_ID_1>0 && ((!(!((abs(lep_ID_0)==11&&lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20))&&((abs(lep_ID_0)==11&&lep_Mtrktrk_atPV_CO_0<0.1&&lep_Mtrktrk_atPV_CO_0>=0)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atPV_CO_1<0.1&&lep_Mtrktrk_atPV_CO_1>=0)))&&!((abs(lep_ID_0)==11&&lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20)))))" -o /eos/user/v/vandergr/sstt_data/Hybrid_Selection/tty.h5
