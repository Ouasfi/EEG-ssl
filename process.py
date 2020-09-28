from preprocessing import *
import numpy as np

def decode_event(event_id):
        stimulus_id = event_id // 10
            condition = event_id % 10
                return stimulus_id, condition

            def get_stim_events(events, stimulus_ids='all', conditions='all'):
                    filtered = []
                        for event in events:
                                    event_id = event[2]
                                            if event_id >= 1000:
                                                            continue

                                                                stimulus_id, condition = decode_event(event_id)

                                                                        if (stimulus_ids == 'all' or stimulus_id in stimulus_ids) and \
                                                                                                (conditions == 'all' or condition in conditions):
                                                                                                                filtered.append(event)

                                                                                                                    return np.array(filtered):
