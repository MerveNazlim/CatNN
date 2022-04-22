#!/bin/bash

unset DISPLAY
echo Running on host `hostname`
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
path="/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v07/v0701/systematics-full/nominal/"
cd /afs/cern.ch/user/v/vandergr/private/DNN/source; source setup.sh;
ttree2hdf5 "$path"mc16a/304014.root "$path"mc16d/304014.root "$path"mc16e/304014.root "$path"mc16a/410081.root "$path"mc16d/410081.root "$path"mc16e/410081.root "$path"mc16a/410408.root "$path"mc16d/410408.root "$path"mc16e/410408.root "$path"mc16a/364242.root "$path"mc16a/364243.root "$path"mc16a/364244.root "$path"mc16a/364245.root "$path"mc16a/364246.root "$path"mc16a/364247.root "$path"mc16a/364248.root "$path"mc16a/364249.root "$path"mc16d/364242.root "$path"mc16d/364243.root "$path"mc16d/364244.root "$path"mc16d/364245.root "$path"mc16d/364246.root "$path"mc16d/364247.root "$path"mc16d/364248.root "$path"mc16d/364249.root "$path"mc16e/364242.root "$path"mc16e/364243.root "$path"mc16e/364244.root "$path"mc16e/364245.root "$path"mc16e/364246.root "$path"mc16e/364247.root "$path"mc16e/364248.root "$path"mc16e/364249.root "$path"mc16a/410560.root "$path"mc16d/410560.root "$path"mc16e/410560.root "$path"mc16a/410560.root "$path"mc16d/410560.root "$path"mc16e/410560.root "$path"mc16a/342284.root  "$path"mc16a/342285.root  "$path"mc16d/342284.root  "$path"mc16d/342285.root  "$path"mc16e/342284.root  "$path"mc16e/342285.root "$path"mc16a/346799_AF.root  "$path"mc16d/346799_AF.root "$path"mc16e/346799_AF.root "$path"mc16a/346678_AF.root "$path"mc16d/346678_AF.root "$path"mc16e/346678_AF.root --branch-regex "nJets_OR|jet_pseudoscore_DL1r0|jet_pseudoscore_DL1r1|jet_pseudoscore_DL1r2|DRll01|sumPsbtag|HT_jets|jet_pt0_nofwd|met_met|MtLepMet|HT_lep|quadlep_type|eventNumber|totalEventsWeighted|RunYear|weight_pileup|jvtSF_customOR|bTagSF_weight_DL1r_77|weight_mc|xs|dilep_type|Mll01|nJets_OR_DL1r_77|nJets_OR_DL1r_70|nJets_OR_DL1r_60|nJets_OR_DL1r_85" -s "custTrigMatch_TightElMediumMuID_FCLooseIso_SLTorDLT&&dilep_type && ((abs(lep_ID_0) == 13 && lep_isMedium_0 && lep_isolationFCLoose_0 && passPLIVTight_0 ) || (abs(lep_ID_0) == 11 && lep_isolationFCLoose_0 && lep_isTightLH_0 && lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 && lep_ambiguityType_0 == 0 && passPLIVTight_0)) && ((abs(lep_ID_1) == 13 && lep_isMedium_1 && lep_isolationFCLoose_1 && passPLIVTight_1 ) || (abs(lep_ID_1) == 11 && lep_isolationFCLoose_1 && lep_isTightLH_1 && lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 && lep_ambiguityType_1 == 0 && passPLIVTight_1 )) && lep_Pt_0*1e-3>20 && lep_Pt_1*1e-3>20 && nJets_OR>=2 && nJets_OR_DL1r_77>=1 && lep_ID_0*lep_ID_1>0 && (dilep_type&&(!(!((abs(lep_ID_0)==11&&lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20))&&((abs(lep_ID_0)==11&&lep_Mtrktrk_atPV_CO_0<0.1&&lep_Mtrktrk_atPV_CO_0>=0)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atPV_CO_1<0.1&&lep_Mtrktrk_atPV_CO_1>=0)))&&!((abs(lep_ID_0)==11&&lep_Mtrktrk_atConvV_CO_0<0.1&&lep_Mtrktrk_atConvV_CO_0>=0&&lep_RadiusCO_0>20)||(abs(lep_ID_1)==11&&lep_Mtrktrk_atConvV_CO_1<0.1&&lep_Mtrktrk_atConvV_CO_1>=0&&lep_RadiusCO_1>20))))" -o /eos/user/v/vandergr/Other.h5
