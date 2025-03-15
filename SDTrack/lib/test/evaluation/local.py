from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.network_path = '/data/users/shanym/SDTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/users/shanym/SDTrack'
    settings.result_plot_path = '/data/users/shanym/SDTrack/output/test/result_plots'
    settings.results_path = '/data/users/shanym/SDTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/users/shanym/SDTrack/output'
    settings.segmentation_path = '/data/users/shanym/SDTrack/output/test/segmentation_results'
    settings.eotb_path = '/data/dataset/FE108_3C/test'
    settings.visevent_path = '/data/dataset/VisEvent/test/'
    settings.felt_path = '/data/dataset/FELT/test/'
    return settings

