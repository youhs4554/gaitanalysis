from utils.preprocessing import PatientLocalizer, COPAnalyizer, Worker
import opts
from preprocess.darknet.python.extract_bbox import set_gpu

opt = opts.parse_opts()

# patient localizer & interval selector
set_gpu(opt.device_yolo)

localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)

interval_selector = COPAnalyizer(opt.meta_home, opt.fps)
worker = Worker(localizer, interval_selector, opt)