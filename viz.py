import pm4py

if __name__ == "__main__":
    log = pm4py.read_xes('lstm_log.xes')

    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    pm4py.view_dfg(dfg, start_activities, end_activities)