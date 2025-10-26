import numpy as np
import os
from utils_ridge.utils import load_data_bling, load_data, zscore

class ResponseUtils:
    def __init__(self):
        self.participant_names = ["COL", "GFW", "KOP", "TYE", "ZEK", "JUS"]
        self.stories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy', 'life', 'myfirstdaywiththeyankees', 'odetostepfather', 'souls', 'undertheinfluence']
        self.test_story = ['wheretheressmoke']
        self.trim={'zh':8, 'en':8}
        self.file_suffix = {'zh':'Audio_zh.hf5','en':'Audio_en.hf5'}
        self.file_suffix_final = {'zh':'Audio_zh.hdf','en':'Audio_en.hdf'}

    def zscore(self, a, mean=None, std=None, return_info=False):
        # a is [TRs x voxels]
        EPSILON = 0.000000001
        if type(mean) != np.ndarray:
            mean = np.nanmean(a, axis=0)
        if type(std) != np.ndarray:
            std = np.nanstd(a, axis=0)
        if not return_info:
            return (a - np.expand_dims(mean, axis=0)) / (np.expand_dims(std, axis=0) + EPSILON)
        return (a - np.expand_dims(mean, axis=0)) / (np.expand_dims(std, axis=0) + EPSILON), mean, std

    def get_resp_train(self, DATA_TRAIN_DIR, subject, language, stack = True, vox = None):
        """loads response data
        num_trs x num_voxels
        """
        subject_idx = self.participant_names.index(subject) + 1
        resp = {}
        trim = 5
        for story in self.stories:
            #resp[story] = load_data_bling(os.path.join(DATA_TRAIN_DIR,f'YYYYMMDD{subject}',f'{story}{self.file_suffix[language]}'))[5+5:-5-self.trim[language],:]
            if 'COLL' in subject and 'en' in language:
                resp[story] = self.zscore(np.nan_to_num(load_data_bling(os.path.join(DATA_TRAIN_DIR,f'YYYYMMDD{subject}',f'{story}.hf5'))[5+trim:-self.trim[language]-5,:]))
            else:
                resp[story] = self.zscore(np.nan_to_num(load_data_bling(os.path.join(DATA_TRAIN_DIR,f'YYYYMMDD{subject}',f'{story}{self.file_suffix_final[language]}'))[5+trim:-self.trim[language]-5,:]))
            if vox is not None:
                resp[story] = resp[story][:, vox]
        #resp = {story: self.zscore(story_resp) for story, story_resp in resp.items()}
        resp = {story: story_resp for story, story_resp in resp.items()}
        if stack:
            return np.vstack([resp[story] for story in self.stories]), self.stories
        else:
            return resp, self.stories
        
    def load_subject_fMRI(self, fdir, subject, modality):
        fname_tr5 = os.path.join(fdir, 'subject{}_{}_fmri_data_trn.hdf'.format(subject, modality))
        trndata5 = load_data(fname_tr5)
        print(trndata5.keys())

        fname_te5 = os.path.join(fdir, 'subject{}_{}_fmri_data_val.hdf'.format(subject, modality))
        tstdata5 = load_data(fname_te5)
        print(tstdata5.keys())
        trnkeys = ['story_01', 'story_02', 'story_03', 'story_04', 'story_05', 'story_06', 'story_08', 'story_09', 'story_10']
        
        trim = 5
        zRresp = np.vstack([self.zscore(trndata5[story][5+trim:-trim-5,:]) for story in trnkeys])
        zPresp = np.vstack([self.zscore(tstdata5[story][0][5+trim:-trim-5,:]) for story in tstdata5.keys()])
        return zRresp, zPresp

    def get_resp_test(self, DATA_TEST_DIR, subject, language, single_trial: bool = True, trim_start: int = 10, trim_end: int = 10):
        """loads response data
        num_trs x num_voxels
        """
        resp = {}
        trim = 5
        #resp = load_data_bling(os.path.join(DATA_TEST_DIR, f'YYYYMMDD{subject}',f'{self.test_story[0]}{self.file_suffix[language]}'))[5+5:-5-self.trim[language],:]
        if 'COLL' in subject and 'en' in language:
                resp[story] = load_data_bling(os.path.join(DATA_TEST_DIR,f'YYYYMMDD{subject}',f'{story}.hf5'))
        else:
            resp = self.zscore(np.nan_to_num(load_data_bling(os.path.join(DATA_TEST_DIR, f'YYYYMMDD{subject}',f'{self.test_story[0]}{self.file_suffix_final[language]}'))[5+trim:-self.trim[language]-5,:]))
        #resp = self.zscore(resp)
        return resp
