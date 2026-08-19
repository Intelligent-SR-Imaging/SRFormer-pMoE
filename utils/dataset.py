from dataset_baseline import Dataset_baseline


class Dataset_3dnt_augment_in_cache_extend_baseline(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs()
        super().augment_in_cache_swin(super().get_img_gt_in)

    def get_data(self, i):
        return super().get_data_from_augment_in_cache(i)

class Dataset_3dnt_augment_eachit_in_cache_extend_baseline_VSR_swin(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs()

    def get_data(self, i):
        # return super().get_data_from_augment_in_cache(i)
        return super().augment_batch_swin(super().get_img_gt_in)
