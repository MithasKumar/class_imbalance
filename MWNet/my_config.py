N: 3
# adv_pen: null
# adv_pen_perc: 50
# beta1_cls: 0.9
# beta1_disc: 0.5
# beta1_gen: 0.5
# beta1_src: 0.9
# beta1_tar: 0.9
# beta2_cls: 0.999
# beta2_disc: 0.999
# beta2_gen: 0.999
# beta2_src: 0.999
# beta2_tar: 0.999
bs: 32
# classifier_loss_func: WFL
classify_every_epoch: true
# clp_l: -0.01
# clp_u: 0.01
# cls_pen: null
# critic_iters: -1
data_norm_lvl: Local
data_norm_mode: null
data_type: Raw
# dec_layers: null
# dec_layers_info: null
def_f1: 256
def_f2: 128
def_f3: 64
enc_layers: 3
enc_layers_info: '[256,128]'
ensemble: false
epochs: 501
f1: null
f1d: 512
f2: null
f2d: 256
f3: null
f3d: 128
freeze_enc_tar_till: null
gan_mode: WGAN
gan_sl: 1
gen_iters: 1
gp_lambda: 11
impose_scalers: false
jobid: TEST_ID
lam: '0.1'
ld: 64
load_id: null
load_id2: null
load_mn2: Classifier
load_mn2_tar: Classifier
load_mn_src_dec: Decoder_src
load_mn_src_enc: Encoder_src
load_mn_tar_enc: Encoder_tar
load_pre_train_dec: false
load_pre_train_enc: All_layers
lpchz_mode: GP
lr_disc: 0.0001
lr_src_cls: 0.0001
lr_src_enc_cls: 0.0001
lr_src_enc_dec: 0.0005
lr_tar_cls: 0.0001
lr_tar_enc_dec: 0.0005
lr_tar_gen: 0.0001
mn1_src: Encoder_src
mn1_tar: Encoder_tar
mn2_src: Decoder_src
mn2_tar: Decoder_tar
mn3: Discriminator
mn4: Classifier
model_mode: null
only_deterministic: false
opt_cls: Adam
opt_ed_src: Adam
opt_ed_tar: Adam
opt_gd: Adam
opt_type: max_min
prnt_lr_updates: true
prnt_processing_info: false
rand_seed: -1
recons: Win_Zeros_4
recons_loss_func: MSE
save_embeds: false
schdl_opt_cls: null
schdl_opt_cls_enc: null
schdl_opt_cls_tar: null
schdl_opt_dec_src: null
schdl_opt_dec_tar: null
schdl_opt_disc: null
schdl_opt_enc_src: null
schdl_opt_enc_tar: null
schdl_opt_gen: null
seed_upsampler: false
target_recons_wait_epochs: 0
target_tune_classifier_epochs: 5
temp_thresh: 2001
threshold_penalty_weight: 0.1
threshold_selection_criterion: spec_sens_norm
train_mode: Source_pipeline
train_tar_rat: null
tsne: Both
tsne_labels: 1
tsql: null
tune_tar_with_cls: false
upsample: null
use_loader_sampler: false
want_probab_plots: false
# wd_src_cls: 0
# wd_src_enc_dec: 0
# wd_tar_cls: 0
# wfl_alpha: 0.07
# wfl_gamma: 2
window_classifier_configs: '[256,128]'