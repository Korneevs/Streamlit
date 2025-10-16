import sys
sys.path.append('../Lib')
import warnings
warnings.filterwarnings("ignore")
from media.TV_analyzer.PLATVA_metric_dataset import PLATVA_metric_dataset
from media.TV_analyzer.PLATVA_results import PLATVA_results
from media.TV_analyzer.PLATVA_TVNONTV_users_maker import PlatvaTVNONTVUsersMaker


class PLATVA(PLATVA_results, PLATVA_metric_dataset, PlatvaTVNONTVUsersMaker):
    pass
