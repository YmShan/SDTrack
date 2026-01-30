from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/dataset/got10k_lmdb'
    settings.got10k_path = '/data/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/dataset/itb'
    settings.lasot_extension_subset_path_path = '/data/dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/dataset/lasot_lmdb'
    settings.lasot_path = '/data/dataset/lasot'
    settings.network_path = '/data/users/xxx/SDTrack-Event/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/dataset/nfs'
    settings.otb_path = '/data/dataset/otb'
    settings.prj_dir = '/data/users/xxx/SDTrack-Event'
    settings.result_plot_path = '/data/users/xxx/SDTrack-Event/output/test/result_plots'
    settings.results_path = '/data/users/xxx/SDTrack-Event/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/users/xxx/SDTrack-Event/output'
    settings.segmentation_path = '/data/users/xxx/SDTrack-Event/output/test/segmentation_results'
    settings.tc128_path = '/data/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/dataset/trackingnet'
    settings.uav_path = '/data/dataset/uav'
    settings.vot18_path = '/data/dataset/vot2018'
    settings.vot22_path = '/data/dataset/vot2022'
    settings.vot_path = '/data/dataset/VOT2019'
    settings.youtubevos_dir = ''
    settings.eotb_path = '/data/dataset/FE108/test'
    settings.visevent_path = '/data/dataset/VisEvent/test/'
    settings.felt_path = '/data/dataset/FELT/test/'
    return settings

