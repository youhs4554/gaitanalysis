from utils.preprocessing import PatientLocalizer, COPAnalyizer, Worker
import opts

opt = opts.parse_opts()

localizer = PatientLocalizer(darknet_api_home=opt.darknet_api_home)

interval_selector = COPAnalyizer(opt.meta_home, opt.fps)
worker = Worker(localizer, interval_selector, opt)