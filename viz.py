from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_factory


log_path = "lstm_log.xes"
log = xes_importer.apply(log_path)

event_log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

dfg = dfg_factory.apply(event_log)

dfg_gviz = dfg_visualizer.apply(dfg, log=event_log, variant=dfg_visualizer.Variants.FREQUENCY)
dfg_visualizer.view(dfg_gviz)