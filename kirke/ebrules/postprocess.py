
from typing import Dict, List, Tuple


# pylint: disable=too-few-public-methods
class AdjacentLineMerger:

    def __init__(self):
        pass

    # merge adjacent positive samples
    # remove negative samples
    # pylint: disable=too-many-locals, no-self-use
    def apply_post_process(self,
                           prob_samples: List[Tuple[float, Dict]],
                           text: str) \
                           -> List[Tuple[float, Dict]]:
        grouped_samples = []  # type: List[List[Tuple[float, Dict]]]
        current_group = []  # type: List[Tuple[float, Dict]]
        for prob, sample in prob_samples:
            if prob >= 0.5:
                if not current_group:
                    grouped_samples.append(current_group)
                current_group.append((prob, sample))
            else:
                if current_group:
                    current_group = []
                # we skip non-addresses

        # now merge groups
        result = []  # type: List[Tuple[float, Dict]]
        for grouped_sample in grouped_samples:
            first_prob, first_sample = grouped_sample[0]
            unused_last_prob, last_sample = grouped_sample[-1]
            max_prob = first_prob
            for tmp_prob, unused_tmp_sample in grouped_sample[1:]:
                if tmp_prob > max_prob:
                    max_prob = tmp_prob


            start = first_sample['start']
            end = last_sample['end']
            merged_sample = {'sample_type': 'line',
                             'start': start,
                             'end': end,
                             'line_seq': first_sample['line_seq'],
                             'text': text[start:end],
                             'prev_n_words': first_sample['prev_n_words'],
                             'post_n_words': last_sample['post_n_words']}
            result.append((max_prob, merged_sample))
        return result
