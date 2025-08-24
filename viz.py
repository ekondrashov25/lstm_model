import pm4py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_main_events_only(df_filtered): # visualize only the main training events
    main_events = ['ForwardPass', 'BackwardPass', 'WeightsUpdate', 'GradientClipping', 
                  'GateSummary', 'Validation', 'EpochEnd']
    df_main = df_filtered[df_filtered['concept:name'].isin(main_events)]
    
    df_main = df_main.sort_values('time:timestamp')
    chunk_size = 50
    df_main['new_case_id'] = (df_main.index // chunk_size).astype(str)
    df_main['case:concept:name'] = 'trace_' + df_main['new_case_id']
    
    log_main = pm4py.convert_to_event_log(df_main)
    dfg_main, start_main, end_main = pm4py.discover_dfg(log_main)
    
    logging.info(f"Main events only")
    logging.info(f"Events: {len(log_main)}")
    logging.info(f"Start activities: {start_main}")
    logging.info(f"End activities: {end_main}")
    
    pm4py.view_dfg(dfg_main, start_main, end_main)

def visualize_gate_events_only(df_filtered): # visualize only the gate-related events
    gate_events = ['Gate_Forget', 'Gate_Input', 'Gate_Output', 'Gate_Candidate', 
                  'GateGradient_Forget', 'GateGradient_Input', 'GateGradient_Output', 'GateGradient_Candidate']
    df_gates = df_filtered[df_filtered['concept:name'].isin(gate_events)]
    
    df_gates = df_gates.sort_values('time:timestamp')
    chunk_size = 50
    df_gates['new_case_id'] = (df_gates.index // chunk_size).astype(str)
    df_gates['case:concept:name'] = 'trace_' + df_gates['new_case_id']
    
    log_gates = pm4py.convert_to_event_log(df_gates)
    dfg_gates, start_gates, end_gates = pm4py.discover_dfg(log_gates)
    
    logging.info(f"Gate events only")
    logging.info(f"Events: {len(log_gates)}")
    logging.info(f"Start activities: {start_gates}")
    logging.info(f"End activities: {end_gates}")
    
    pm4py.view_dfg(dfg_gates, start_gates, end_gates)

def visualize_weight_events_only(df_filtered): # visualize only the weight-related events
    weight_events = ['WeightGradient_W_f', 'WeightGradient_W_i', 'WeightGradient_W_c', 
                    'WeightGradient_W_o', 'WeightGradient_W_hy',
                    'WeightUpdate_W_f', 'WeightUpdate_W_i', 'WeightUpdate_W_c', 
                    'WeightUpdate_W_o', 'WeightUpdate_W_hy']
    df_weights = df_filtered[df_filtered['concept:name'].isin(weight_events)]
    
    df_weights = df_weights.sort_values('time:timestamp')
    chunk_size = 50
    df_weights['new_case_id'] = (df_weights.index // chunk_size).astype(str)
    df_weights['case:concept:name'] = 'trace_' + df_weights['new_case_id']
    
    log_weights = pm4py.convert_to_event_log(df_weights)
    dfg_weights, start_weights, end_weights = pm4py.discover_dfg(log_weights)
    
    logging.info(f"Weight events only")
    logging.info(f"Events: {len(log_weights)}")
    logging.info(f"Start activities: {start_weights}")
    logging.info(f"End activities: {end_weights}")
    
    pm4py.view_dfg(dfg_weights, start_weights, end_weights)

if __name__ == "__main__":
    log = pm4py.read_xes('lstm_log.xes')
    
    df = pm4py.convert_to_dataframe(log)
    
    logging.info("All event types in the log:")
    event_types = df['concept:name'].unique()
    for event_type in sorted(event_types):
        count = len(df[df['concept:name'] == event_type])
        logging.info(f"  {event_type}: {count} occurrences")
    
    # remove metadata events due to error in pm4py while converting to event log
    metadata_events = ['DatasetInfo', 'ModelArchitecture']
    df_filtered = df[~df['concept:name'].isin(metadata_events)]
    logging.info(f"After removing metadata events: {len(df_filtered)} events")

    # remove GradientClipping events due to the large number of events
    gradient_clipping_mask = df_filtered['concept:name'] == 'GradientClipping'
    gradient_clipping_indices = df_filtered[gradient_clipping_mask].index
    if len(gradient_clipping_indices) > 100: 
        sample_indices = gradient_clipping_indices[::10]
        df_filtered = df_filtered[~gradient_clipping_mask]
        df_filtered = pd.concat([df_filtered, df.loc[sample_indices]])
        df_filtered = df_filtered.sort_index()
        logging.info(f"After sampling GradientClipping: {len(df_filtered)} events")

    
    show_detailed = True
    
    if not show_detailed:
        main_events = ['ForwardPass', 'BackwardPass', 'WeightsUpdate', 'GradientClipping', 
                      'GateSummary', 'Validation', 'EpochEnd']
        df_filtered = df_filtered[df_filtered['concept:name'].isin(main_events)]
        logging.info(f"After filtering to main events: {len(df_filtered)} events")
    
    logging.info("Event types before conversion to event log:")
    final_event_types = df_filtered['concept:name'].unique()
    for event_type in sorted(final_event_types):
        count = len(df_filtered[df_filtered['concept:name'] == event_type])
        logging.info(f"  {event_type}: {count} occurrences")
    
    logging.info(f"Number of traces: {df_filtered['case:concept:name'].nunique()}")
    logging.info(f"Trace names: {df_filtered['case:concept:name'].unique()}")
    
    df_filtered = df_filtered.sort_values('time:timestamp')
    
    chunk_size = 50  # Events per trace
    df_filtered['new_case_id'] = (df_filtered.index // chunk_size).astype(str)
    
    df_filtered['case:concept:name'] = 'trace_' + df_filtered['new_case_id']
    
    logging.info(f"Created {df_filtered['case:concept:name'].nunique()} new traces with {chunk_size} events each")
    
    log_filtered = pm4py.convert_to_event_log(df_filtered)
    
    logging.info(f"Original log events: {len(log)}")
    logging.info(f"Filtered log events: {len(log_filtered)}")
    logging.info(f"Removed metadata events: {metadata_events}")
    if len(gradient_clipping_indices) > 100:
        logging.info(f"Sampled GradientClipping events: kept {len(sample_indices)} out of {len(gradient_clipping_indices)}")
    
    dfg, start_activities, end_activities = pm4py.discover_dfg(log_filtered)
    
    logging.info(f"Start activities: {start_activities}")
    logging.info(f"End activities: {end_activities}")
    
    pm4py.view_dfg(dfg, start_activities, end_activities)
    
    logging.info("different detalization levels of events")
    
    # uncomment the following lines to see different visualizations
    # visualize_main_events_only(df_filtered)
    # visualize_gate_events_only(df_filtered)
    # visualize_weight_events_only(df_filtered)