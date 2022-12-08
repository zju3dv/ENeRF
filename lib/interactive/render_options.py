# this is the render option class, just a dot dict
# it should control all modifiable render options through the imgui options

from lib.utils.base_utils import DotDict
from lib.config import cfg


opt = DotDict()

# -----------------------------------------------------------------------------
# * Interactive Rendering Related
# -----------------------------------------------------------------------------
opt.fps_cnter_int = 1  # update fps per 0.5 seconds
opt.render_level = 1  # indexing rendering scale
opt.type = 0  # indexing rendering scale
opt.type_mapping = ['pred', 'depth', 'seg', 'bbox']

if cfg.test_dataset.scene == 'taekwondo' or cfg.test_dataset.scene == 'walking':
    opt.window_hw = [320, 640]
elif 'cook' in cfg.test_dataset.scene or 'flame' in cfg.test_dataset.scene or 'coffee' in cfg.test_dataset.scene:
    opt.window_hw = [448, 640]
else:
    opt.window_hw = [512, 512]
# opt.window_hw = 512, 512
opt.font_filepath = 'lib/interactive/fonts/Caskaydia Cove Nerd Font Complete.ttf'
# opt.lock_fxfy = True
opt.autoplay = True

opt.smoothing_term = 0.1
