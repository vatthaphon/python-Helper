import os.path
import pickle
import sys


ROOTPATH_g = (os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")).split("/Data")[0]


sys.path.append(ROOTPATH_g + "/Data/Work/UtilSrcCode/python-Helper")
from AV_helper import AV_norm_path, AV_delFile


if __name__ == "__main__":
    ########## We set the number of cores by running this python file at Ubuntu.

    # Pre
    # study: Mon afternoon, Tue afternoon, Wed afternoon, Fri. morning
    # "yok"
    # study: Mon, Tue afternoon, Wed


    # all_clients_N_cores_g = {"vattha_mac": 1, "yok": 11, "vattha": 11, "pre": 11, "thanaphon": 11}
    all_clients_N_cores_g = {
    "vattha_mac": 1, 
    
    "yok": 11,
    # "yok": 3, 
    # "yok": 5, 

    # "vattha": 1, 
    # "vattha": 5, 
    # "vattha": 9, 
    "vattha": 11, 

    # "pre": 1, 
    # "pre": 3,     
    # "pre": 5,     
    "pre": 11, 

    "nam": 11, 
    # "nam": 5, 
    # "nam": 3,    

    "thanaphon": 11
    # "thanaphon": 5
    # "thanaphon": 3
    }

    python_Helper_l = ROOTPATH_g + "/Data/Work/UtilSrcCode/python-Helper"

    with open(python_Helper_l + "/N_cores_g_profiles", "wb+") as fp:
    	pickle.dump(all_clients_N_cores_g, fp)
    	fp.close()        

    ########## We choose which servers are used for ML
    is_used_for_ML = {
    # "vattha": True, 
    "vattha": False, 
    "pre": True, 
    # "pre": False, 	
    "yok": True, 
    # "yok": False, 
    "nam": True, 
    # "nam": False,     
    "thanaphon": True
    # "thanaphon": False
    }


    if is_used_for_ML["vattha"]:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_vattha_nonactive"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_vattha_active"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()        
    else:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_vattha_active"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_vattha_nonactive"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()

    if is_used_for_ML["pre"]:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_pre_nonactive"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_pre_active"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()                
    else:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_pre_active"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_pre_nonactive"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()        

    if is_used_for_ML["yok"]:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_yok_nonactive"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_yok_active"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()                
    else:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_yok_active"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_yok_nonactive"), "wb+") as fp:
            tmp_l = 0
            pickle.dump([tmp_l], fp)
            fp.close()        

    if is_used_for_ML["thanaphon"]:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_thanaphon_nonactive"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_thanaphon_active"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()                
    else:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_thanaphon_active"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_thanaphon_nonactive"), "wb+") as fp:
            tmp_l = 0
            pickle.dump([tmp_l], fp)
            fp.close()        

    if is_used_for_ML["nam"]:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_nam_nonactive"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_nam_active"), "wb+") as fp:
            pickle.dump([0], fp)
            fp.close()                
    else:
        AV_delFile(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_nam_active"))

        with open(AV_norm_path("D:\\vattha\\Data\\Work\\FIRST\\EEG\\DataResults\\MUSE\\MW\\muse_ml_job_nam_nonactive"), "wb+") as fp:
            tmp_l = 0
            pickle.dump([tmp_l], fp)
            fp.close() 


