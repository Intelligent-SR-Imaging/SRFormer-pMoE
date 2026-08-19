from model.SRFormer_pMoE import SRFormer_pMoE


def get_model(opt):
    return eval(opt)

def get_model_with_opt(get_model, opt):
    getmodel = eval(get_model)
    return getmodel(opt)

def getSRFormer_pMoE(opt):
    return SRFormer_pMoE(upscale=[opt.zupfactor, opt.upfactor, opt.upfactor],
                         bn= opt.bn,
                         img_size=[opt.input_frame, opt.noise_crop,opt.noise_crop],
                         depth = opt.depths,
                         embed_dim= opt.embed_dim,
                         num_heads=opt.num_heads,
                         expansion_factor=opt.expansion_factor,
                         resi_connection=opt.resi_connection,
                         out_proj= opt.out_proj,
                         split_size= opt.split_size,
                         drop_rate=opt.drop_rate,
                         attn_drop_rate=opt.attn_drop_rate,
                         sgfn_drop=opt.drop_rate,
                         upsampler = opt.upsampler,
                         bayesian=opt.bayesian,
                         moe_attn_cw=opt.moe_attn_cw,
                         moe_attn_sw=opt.moe_attn_sw,
                         moe_ffn_cw=opt.moe_ffn_cw,
                         moe_ffn_sw=opt.moe_ffn_sw,
                         n_shared_experts=opt.n_shared_experts,
                         n_routed_experts=opt.n_routed_experts,
                         n_activated_experts=opt.n_activated_experts,
                         score_func=opt.score_func,
                         route_scale=opt.route_scale,
                         normal_moe_weight=opt.normal_moe_weight,
                         aux_free_loss = opt.aux_free_loss,
                         aux_free_batch=opt.aux_free_batch,
                         aux_free_rate=opt.aux_free_rate,
                         expert_type=opt.expert_type,
                         gate_feature_dim=opt.gate_feature_dim,
                         gate_f_type=opt.gate_f_type,
                         )
