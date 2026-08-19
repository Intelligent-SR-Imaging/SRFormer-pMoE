
class BasicConfig(object):
    def __init__(self):
        self.type = "base"


_C = BasicConfig()

# TODO envs model
_C.env_name = "SRFormer_pMoE_retrain"
_C.arch = "getSRFormer_pMoE"
_C.input_frame = 12  # be modified to even number
_C.output_frame = 12
_C.depths = [6,6,6,6]
_C.embed_dim = 90 # must be multiple of num_heads
_C.num_heads=[6,6,6,6]
_C.expansion_factor = 2
_C.resi_connection = 'SFFB3D'
_C.out_proj = '1conv'
_C.split_size = [[4, 16, 16],[4, 16, 16]]
_C.drop_rate = 0.
_C.attn_drop_rate = 0.
_C.drop_rate = 0.
_C.upsampler = 'pixelshuffle'
_C.bn = False
_C.half = False

_C.moe_attn_cw = False
_C.moe_attn_sw = False
_C.moe_ffn = True
_C.gate_feature_dim = 3
_C.gate_f_type = 'AvgPool'
_C.moe_ffn_cw=True
_C.moe_ffn_sw=True

_C.aux_loss = False
_C.aux_loss_weight = 0.1

_C.aux_free_loss= True
_C.aux_free_batch = 256
_C.aux_free_rate = 0.001
_C.expert_type = 'Expert_conv'
_C.n_shared_experts=2
_C.n_routed_experts=32
_C.n_activated_experts=2
_C.score_func="softmax"
_C.route_scale=1.
_C.channel_map="direct"
_C.normal_moe_weight = False

_C.bayesian = False

_C.perceptual_loss = True
_C.perceptual_loss_xz = False
_C.perceptual_loss_yz = False
_C.perceptual_loss_weight = 0.1

_C.resume = False
_C.resume_path = None
_C.resume_input_frame=_C.input_frame
_C.resume_output_frame=_C.output_frame 
_C.resume_noise_crop= 64
        
_C.device = 'cuda'
_C.device_num = "3,4,5,6"
_C.main_device_num = "0"
_C.main_device_id = 0
_C.device_ids = [0,1,2,3]

# TODO train data
#TODO switch in test/train
_C.train_dir = './data/LLSM/general_struct/train_zup'
# TODO
_C.val_dir = './data/LLSM/general_struct/val_zup'
_C.test_dir = './data/LLSM/general_struct/test_zup'

# TODO
_C.input_dir = ['20230815_Ens_LZ/input_slice_1','20230815_Ens_LZ/input_slice_2',
                '20240111_Factin_LZ_fix/input_high_slice_1','20240111_Factin_LZ_fix/input_high_slice_2',
                '20240111_Factin_LZ_fix/input_high_slice_4_6','20240111_Factin_LZ_fix/input_high_slice_7_9',
                 '2024051617_Tom20_LZ_fix/input_p1_slice_1_3','2024051617_Tom20_LZ_fix/input_p1_slice_7_9',
                '2024051617_Tom20_LZ_fix/input_p1_slice_1_6','2024051617_Tom20_LZ_fix/input_p2_slice_1_3',
                 '20240111_Tom20_LZ_fix/highSNRInput','20240111_Tom20_LZ_fix/highSNRInput_slice_1_3',
                '20240111_Tom20_LZ_fix/lowSNRInput','20240111_Tom20_LZ_fix/lowSNRInput_slice_4_6',
                '20230904_ER_LZ/input_slice_1_3','20230904_ER_LZ/input_slice_2_9',
                '20240410_G3BP1_LZ/input_slice_1','20240410_G3BP1_LZ/input_slice_1_6',
                '20230815_Ens_LZ/input_slice_2_9','20230815_Ens_LZ/input_slice_1_3']
_C.gt_dir = ['20230815_Ens_LZ/gt','20230815_Ens_LZ/gt',
             '20240111_Factin_LZ_fix/gt','20240111_Factin_LZ_fix/gt',
        '20240111_Factin_LZ_fix/gt','20240111_Factin_LZ_fix/gt',
                 '2024051617_Tom20_LZ_fix/gt','2024051617_Tom20_LZ_fix/gt',
             '2024051617_Tom20_LZ_fix/gt', '2024051617_Tom20_LZ_fix/gt',
                 '20240111_Tom20_LZ_fix/gt','20240111_Tom20_LZ_fix/gt',
             '20240111_Tom20_LZ_fix/gt', '20240111_Tom20_LZ_fix/gt',
'20230904_ER_LZ/gt','20230904_ER_LZ/gt',
             '20240410_G3BP1_LZ/gt','20240410_G3BP1_LZ/gt_for_slice_1_6',
             '20230815_Ens_LZ/gt','20230815_Ens_LZ/gt']

_C.zupfactor = 3
_C.upfactor = 2 #TODO
_C.is3D = True
_C.is1D = False
_C.img_type = ".tif"
_C.img_frame_start = 0
_C.img_frame = 0
_C.input_img_neg = True
_C.normal = True

# TODO augment
_C.dataset = 'Dataset_3dnt_augment_eachit_in_cache_extend_baseline'
_C.dataset_val = 'Dataset_3dnt_augment_in_cache_extend_baseline'
# TODO split fg vs bg
_C.rotate = False
_C.rotate90 = True
_C.flip = True
_C.frontground_filter = True # filer for 3dnt n2n is cropped by set center at point in mask
_C.fg_th = 0.1
# TODO
_C.augment_size = 10000  # 1536 // 128 = 400  ans * 2 * 20 = 20000
_C.val_augment_size = 1000
_C.noise_crop = 64
_C.clean_crop = _C.noise_crop * _C.upfactor
_C.crop_depth = _C.input_frame

# TODO training
_C.batch_size = 4
_C.nepoch = 2000  # niter

# Network init weight
_C.init_type = "kaiming_uniform"
_C.init_bn_type = "uniform"
_C.init_gain = 0.2

_C.opti = 'adam'
_C.beta1 = 0.5
_C.scheduler = 'multisteplr'
_C.milestones = [60000, 100000, 135000, 185000]#[40000, 55000, 70000, 85000]
_C.gamma = 0.5
_C.lr = 2.5e-5
_C.min_lr = 1e-8
_C.warmup = True
_C.warmup_iter =10000
_C.warmup_intervel = 10000
_C.upper_loss = 1
#TODO switch in test/train
_C.val_checkpoint = 300
_C.save_checkpoint =5000

# TODO test
_C.save_img = '1.tif'
_C.split_test = 16
_C.split_padding = 16

_C.normal_max = 100
