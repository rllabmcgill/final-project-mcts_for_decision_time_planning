# toy example of the full pipeline
from datasets import fetch_two_voice_species1
from datasets import fetch_three_voice_species1
from analysis import notes_to_midi
import numpy as np


def two_voice_species1_wrap():
    all_ex = fetch_two_voice_species1()
    all_lower_offset = []
    all_upper_rel = []
    all_index = []
    for ii, ex in enumerate(all_ex):
        # skip any "wrong" examples
        if not all(ex["answers"]):
            continue
        all_index.append(ex["name"])
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        # durations not used in first species, leave it alone
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        midi = notes_to_midi(notes)
        cf = ex["cantus_firmus_voice"]

        all_lower_offset.append(list(np.array(midi[1]) - midi[1][-1]))

        upper_rel = list(np.array(midi[0]) - np.array(midi[1]))
        all_upper_rel.append(upper_rel)

    flat_upper_rel = [ddd for dd in all_upper_rel for ddd in dd]
    # these map to intervals wrt bottom voice
    # [-8, -4, -3, 0, 3, 4, 7, 8, 9, 12, 15, 16]
    upper_rel_set = sorted(list(set(flat_upper_rel)))
    upper_rel_map = {v: k for k, v in enumerate(upper_rel_set)}

    flat_lower_offset = [ddd for dd in all_lower_offset for ddd in dd]
    # these are input symbols from bottom_voice, as offsets relative to last note ("key" centered)
    # [-12, -10, -9, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 12]
    lower_offset_set = sorted(list(set(flat_lower_offset)))
    lower_offset_map = {v: k for k, v in enumerate(lower_offset_set)}
    return all_lower_offset, lower_offset_map, upper_rel_map, all_index


def three_voice_species1_wrap():
    all_ex = fetch_three_voice_species1()
    all_index = []
    all_tb = []
    all_upper_rel = []
    all_mid_rel = []
    all_um_rel = []
    all_lower_offset = []
    for ii, ex in enumerate(all_ex):
        # skip any "wrong" examples, and also the short one
        if not all(ex["answers"]) or len(ex["answers"]) < 8:
            continue
        all_index.append(ex["name"])
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        # durations not used in first species, leave it alone
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        midi = notes_to_midi(notes)
        cf = ex["cantus_firmus_voice"]

        all_lower_offset.append(list(np.array(midi[2]) - midi[2][-1]))

        all_mid_rel.append(list(np.array(midi[1]) - np.array(midi[2])))
        all_upper_rel.append(list(np.array(midi[0]) - np.array(midi[2])))
        all_um_rel.append(list(np.array(midi[0]) - np.array(midi[1])))

    flat_upper_rel = [ddd for dd in all_upper_rel for ddd in dd]
    # these map to intervals wrt bottom voice
    # [0, 3, 4, 7, 8, 9, 12, 15, 16, 19, 20, 21, 24, 27, 28]
    upper_rel_set = sorted(list(set(flat_upper_rel)))
    upper_rel_map = {v: k for k, v in enumerate(upper_rel_set)}

    flat_mid_rel = [ddd for dd in all_mid_rel for ddd in dd]
    # these map to intervals wrt bottom voice
    # [0, 3, 4, 7, 8, 9, 12, 15, 16, 19, 20, 24]
    mid_rel_set = sorted(list(set(flat_mid_rel)))
    mid_rel_map = {v: k for k, v in enumerate(mid_rel_set)}

    flat_um_rel = [ddd for dd in all_um_rel for ddd in dd]
    # these map to intervals wrt middle voice
    # [-5, -4, 0, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16]
    um_rel_set = sorted(list(set(flat_um_rel)))
    um_rel_map = {v: k for k, v in enumerate(um_rel_set)}

    flat_lower_offset = [ddd for dd in all_lower_offset for ddd in dd]
    # these are input symbols from bottom_voice, as offsets relative to last note ("key" centered)
    # [-16,-14,-12,-11,-10,-9,-8,-7,-5,-4,-3,-2,-1,0,1,2,3,4,5,7,8,9,10,11,12,14]
    lower_offset_set = sorted(list(set(flat_lower_offset)))
    lower_map = {v: k for k, v in enumerate(lower_offset_set)}

    all_comb = [zip(ur, mr) for ur, mr in zip(all_upper_rel, all_mid_rel)]
    flat_all_comb = [ddd for dd in all_comb for ddd in dd]
    all_comb_set = sorted(list(set(flat_all_comb)))
    return all_lower_offset, all_comb_set, upper_rel_map, mid_rel_map, um_rel_map, lower_map, all_index

if __name__ == "__main__":
    two_voice_species1_wrap()
