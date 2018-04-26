# Author: Kyle Kastner
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# See similar implementation here
# https://github.com/junxiaosong/AlphaZero_Gomoku

# changes from high level pseudo-code in survey
# expand all children, but only rollout one
# section biases to unexplored nodes, so the children with no rollout
# will be explored quickly

import numpy as np
import copy
from shared_puct_mcts import MCTS, MemoizeMutable
from dataset_wrap import three_voice_species1_wrap
from analysis import analyze_three_voices, midi_to_notes

all_l, all_c_set, u_map, m_map, um_map, l_map, all_i = three_voice_species1_wrap()
u_inv_map = {v: k for k, v in u_map.items()}
m_inv_map = {v: k for k, v in m_map.items()}
um_inv_map = {v: k for k, v in um_map.items()}
l_inv_map = {v: k for k, v in l_map.items()}

va_u = [u_map[k] for k in sorted(u_map.keys())]
va_m = [m_map[k] for k in sorted(m_map.keys())]
combs = [(u, m) for u in va_u for m in va_m]
j_map = {(u_inv_map[k1], m_inv_map[k2]): (k1, k2) for k1, k2 in combs}
j_map = {k: v for k, v in j_map.items() if k[0] >= k[1]}
j_map = {k: v for k, v in j_map.items() if k[0] != 0 and k[1] != 0}

#j_map = {(k1, k2): (u_map[k1], m_map[k2]) for k1 in sorted(u_map.keys()) for k2 in sorted(m_map.keys())}
# don't constrain to only groupings found in the dataset
#j_map = {k: v for k, v in j_map.items() if k in all_c_set}

j_inv_map = {v: k for k, v in j_map.items()}
j_acts_map = {k: v for k, v in enumerate(sorted(j_map.keys()))}
j_acts_inv_map = {v: k for k, v in j_acts_map.items()}

class ThreeVoiceSpecies1Manager(object):
    def __init__(self, guide_index, offset_value=None, tonality=None, rollout_limit=1000):
        self.tonality = tonality
        self.guide_trace = all_l[guide_index]

        if offset_value is None:
            # [A - A)
            offset_options = np.arange(45, 57)
            offset_names = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
            all_scale_steps = []
            all_scale_names = []
            # Maj
            # Ionian, technically
            io_scale_steps = np.array([0, 2, 4, 5, 7, 9, 11, 12])
            all_scale_steps.append(io_scale_steps)
            all_scale_names.append("Ionian")
            # min
            # Aeolian, technically
            ae_scale_steps = np.array([0, 2, 3, 5, 7, 8, 10, 12])
            all_scale_steps.append(ae_scale_steps)
            all_scale_names.append("Aolian")
            # Dorian
            do_scale_steps = np.array([0, 2, 3, 5, 7, 9, 10, 12])
            all_scale_steps.append(do_scale_steps)
            all_scale_names.append("Dorian")
            # Mixolydian
            mi_scale_step = np.array([0, 2, 3, 5, 7, 8, 10, 12])
            all_scale_steps.append(mi_scale_step)
            all_scale_names.append("Mixolydian")
            # Phrygian
            ph_scale_steps = np.array([0, 1, 3, 5, 7, 8, 10, 12])
            all_scale_steps.append(ph_scale_steps)
            all_scale_names.append("Phrygian")
            # Lydian
            ly_scale_steps = np.array([0, 2, 4, 6, 7, 9, 11, 12])
            all_scale_steps.append(ly_scale_steps)
            all_scale_names.append("Lydian")
            # Locrian
            lo_scale_step = np.array([0, 1, 3, 5, 6, 8, 10, 12])
            all_scale_steps.append(lo_scale_step)
            all_scale_steps.append("Locrian")

            accidentals = np.inf
            least_accidentals = None
            least_index = -1
            for i, offset in enumerate(offset_options):
                guide_notes = midi_to_notes([np.array(self.guide_trace) + offset])[0]
                ac = sum([1 for gn in guide_notes if "#" in gn or "b" in gn])
                if ac < accidentals:
                    accidentals = ac
                    least_accidentals = offset
                    least_index = i
            self.offset_value = least_accidentals
            self.offset_name = offset_names[least_index]
            print("Setting base note {}".format(self.offset_name))
            min_set_diff = np.inf
            min_set = []
            for n in range(len(all_scale_steps)):
                set_diff = len(set(self.guide_trace) - set(all_scale_steps[n]))
                if set_diff <= min_set_diff:
                    if set_diff < min_set_diff:
                        min_set_diff = set_diff
                        min_set = [n]
                    else:
                        min_set.append(n)
            # use our "preferred" min set / mode
            m = min_set[0]
            print("Auto-setting mode to {}".format(all_scale_names[m]))
            self.mode = all_scale_names[m]
            self.scale_steps = all_scale_steps[m]
            base_scale = all_scale_steps[m] + self.offset_value
        else:
            raise ValueError("non-auto tonality support NYI")
            self.offset_value = offset_value

        self.notes_in_scale = np.array(sorted(list(set([si for s in [base_scale + o for o in [-24, -12, 0, 12, 24]] for si in s]))))
        self.random_state = np.random.RandomState(1999)
        self.rollout_limit = rollout_limit
        self.is_finished = MemoizeMutable(self._is_finished)

    def get_next_state(self, state, action):
        tup_act = j_acts_map[action]
        new_state = [state[0] + [tup_act[0]], state[1] + [tup_act[1]], state[2]]
        return new_state

    def get_action_space(self):
        return list(range(len(j_acts_map.keys())))

    def get_valid_actions(self, state):
        s0 = np.array(state[0])
        s1 = np.array(state[1])
        s2 = np.array(state[2])

        if len(state[0]) == 0:
            # for first notes, keep it pretty open
            va_u = [u_map[k] for k in sorted(u_map.keys())]
            va_m = [m_map[k] for k in sorted(m_map.keys())]
            combs = [(u, m) for u in va_u for m in va_m]
            combs = [(u_inv_map[c[0]], m_inv_map[c[1]]) for c in combs]

            # no voice crossing, m/M2 or unison
            combs = [c for c in combs if abs(c[1] - c[0]) > 2 and c[0] > c[1] and c[1] != 0]

            # remove combinations that violate our previous settings for m/M tonality
            #combs = [c for c in combs
            #         if (c[0] not in disallowed and c[1] not in disallowed)]
            # remove combinations with notes not in the scale
            combs = [c for c in combs
                     if c[0] + self.offset_value + state[2][0] in self.notes_in_scale and
                     c[1] + self.offset_value + state[2][0] in self.notes_in_scale]

            # convert to correct option (intervals)
            # make sure it's a viable action
            comb_acts = [j_acts_inv_map[c] for c in combs if c in j_acts_inv_map]
            va = comb_acts
            return va
        else:
            # start at maximum leap interval of a 4th
            upper = 5
            mid = 5
            # iteratively increase action space, slowly growing the possible actions
            # start by loosening the middle voice, then the upper
            while True:
                va_u = [u_map[k] for k in sorted(u_map.keys())]
                va_m = [m_map[k] for k in sorted(m_map.keys())]
                combs = [(u, m) for u in va_u for m in va_m]
                combs = [(u_inv_map[c[0]], m_inv_map[c[1]]) for c in combs]

                state_len = len(state[0])
                state_len = min(state_len, len(state[2]) - 1)

                # heavily constrain top voice, no leap greater than upper
                combs = [c for c in combs if abs((c[0] + state[2][state_len]) - (state[0][state_len - 1] + state[2][state_len - 1])) <= upper]

                # heavily constrain top voice, no leap greater than mid
                combs = [c for c in combs if abs((c[1] + state[2][state_len]) - (state[1][state_len - 1] + state[2][state_len - 1])) <= mid]

                # no voice crossing, m/M2 or unison
                combs = [c for c in combs if abs(c[1] - c[0]) > 2 and c[0] > c[1] and c[1] != 0]

                # remove combinations with notes not in the scale
                combs = [c for c in combs
                         if c[0] + self.offset_value + state[2][state_len] in self.notes_in_scale and
                         c[1] + self.offset_value + state[2][state_len] in self.notes_in_scale]

                # convert to correct option (intervals)
                # make sure it's a viable action
                comb_acts = [j_acts_inv_map[c] for c in combs if c in j_acts_inv_map]
                va = comb_acts
                # try to have a few options
                if len(va) >= 3:
                    break
                else:
                    # no leaps of greater than a 6th
                    if mid < 9:
                        # grow mid action space by 1
                        mid += 1
                    elif mid == 9 and upper < 9:
                        # grow upper action space by 1
                        upper += 1
                    elif mid == 9 and upper == 9:
                        # maximum action space, no choice but to bail
                        break
            return va

    def get_init_state(self):
        top = []
        mid = []
        bot = self.guide_trace
        return copy.deepcopy([top, mid, bot])

    def _rollout_fn(self, state):
        return self.random_state.choice(self.get_valid_actions(state))

    def _score(self, state):
        s0 = np.array(state[0])
        s1 = np.array(state[1])
        s2 = np.array(state[2])
        bot = s2 + self.offset_value
        mid = bot[:len(s1)] + s1
        top = bot[:len(s0)] + s0
        smooth_s0 = 1. / np.sum(np.abs(np.diff(top)))
        smooth_s1 = 1. / np.sum(np.abs(np.diff(mid)))
        unique_max = 1. / float(len(np.where(top == np.max(top))[0]))
        unique_count = 1. / float(len(set(s0)))
        return smooth_s0# + smooth_s1

    def rollout_from_state(self, state):
        s = state
        w, sc, e = self.is_finished(state)
        if e:
            if w == -1:
                return -1.
            elif w == 0:
                return sc
            else:
                return self._score(s)

        c = 0
        while True:
            a = self._rollout_fn(s)
            s = self.get_next_state(s, a)

            w, sc, e = self.is_finished(s)
            c += 1
            if e:
                if w == -1:
                    return -1
                elif w == 0:
                    return sc
                else:
                    return self._score(s)

            if c > self.rollout_limit:
                return 0.

    def _is_finished(self, state):
        if len(state[0]) != len(state[1]):
            raise ValueError("Something bad in is_finished")

        if len(self.get_valid_actions(state)) == 0:
            return -1., -1., True

        if len(state[0]) == 0:
            # nothing happened yet
            return 0, 0., False

        ## only grade it at the end?
        #if len(state[0]) != len(state[2]):
        #    return -1, False

        ns0 = state[0] + [0] * (len(state[2]) - len(state[0]))
        ns1 = state[1] + [0] * (len(state[2]) - len(state[1]))
        # the state to check
        s_l = [ns0, ns1, state[2]]

        s = np.array(s_l)
        s[2, :] += self.offset_value
        s[0, :] += s[2, :]
        s[1, :] += s[2, :]

        parts = s
        durations = [['4'] * len(p) for p in parts]
        key_signature = "C"
        time_signature = "4/4"
        # minimal check during rollout
        aok = analyze_three_voices(parts, durations, key_signature, time_signature,
                                   species="species1_minimal", cantus_firmus_voices=[2])

        if len(aok[1]["False"]) > 0:
            first_error = aok[1]["False"][0]
        else:
            first_error = np.inf

        if len(state[0]) < len(state[2]):
            # error is out of our control (in the padded notes)
            if first_error > (len(state[0]) - 1):
                return 0, 0., False
            else:
                # made a mistake
                return 0, -1. + len(state[0]) / float(len(state[2])), True
        elif aok[0]:
            return 1, 1., True
        else:
            return -1, -1., True


if __name__ == "__main__":
    import time
    from visualization import pitches_and_durations_to_pretty_midi
    from visualization import plot_pitches_and_durations
    from analysis import fixup_parts_durations
    from analysis import intervals_from_midi

    all_parts = []
    all_durations = []
    mcts_random = np.random.RandomState(1110)
    for guide_idx in range(len(all_l)):
        tvsp1m = ThreeVoiceSpecies1Manager(guide_idx)
        mcts = MCTS(tvsp1m, n_playout=1000, random_state=mcts_random)
        resets = 0
        n_valid_samples = 0
        valid_state_traces = []
        temp = 1.
        noise = True
        exact = True
        while True:
            if n_valid_samples >= 1:
                print("Got a valid sample")
                break
            resets += 1
            if resets > 10:
                temp = 1E-3
                noise = False
            elif resets > 15:
                exact = True
            state = mcts.state_manager.get_init_state()
            winner, score, end = mcts.state_manager.is_finished(state)
            states = [state]

            while True:
                if not end:
                    print("guide {}, step {}, resets {}".format(guide_idx, len(states[-1][0]), resets))
                    if not exact:
                        a, ap = mcts.sample_action(state, temp=temp, add_noise=noise)
                    else:
                        a, ap = mcts.get_action(state)

                    if a is None:
                        print("Ran out of valid actions, stopping early at step {}".format(len(states)))
                        print("No actions")
                        end = True

                    if not end:
                        for i in mcts.root.children_.keys():
                            print(i, mcts.root.children_[i].__dict__)
                            print("")
                        mcts.update_tree_root(a)
                        state = mcts.state_manager.get_next_state(state, a)
                        states.append(state)
                        print(state)
                        winner, score, end = mcts.state_manager.is_finished(state)
                        if len(states[-1][0]) == (len(states[-1][2]) - 1):
                            # do the final chord manually
                            end = True
                if end:
                    print(states[-1])
                    mcts.reconstruct_tree()

                    # used to finalize partials
                    # add an ending coda / chord
                    # musical punctuation
                    poss = [0, 7, 12, 19, 24]
                    if 3 in mcts.state_manager.scale_steps:
                        poss += [3, 15, 27]
                    else:
                        poss += [4, 16, 28]
                    possible_ends = [(c0, c1) for c0 in poss for c1 in poss]
                    possible_ends = [(c[0], c[1]) for c in possible_ends if c[0] >= c[1]]

                    min_diff = np.inf
                    min_idx = -1
                    tm1 = states[-1][0][-1]
                    mm1 = states[-1][1][-1]
                    bm1 = states[-1][2][len(states[-1][0]) - 1]
                    bm0 = states[-1][2][len(states[-1][0])]
                    # find the chord combination which minimizes movement in the upper voices, and use it
                    for ii, pe in enumerate(possible_ends):
                        diff = abs((tm1 + bm1) - (bm0 + pe[0])) + abs((mm1 + bm1) - (bm0 + pe[1]))
                        if diff < min_diff:
                            min_diff = diff
                            min_idx = ii

                    states[-1][0].append(possible_ends[min_idx][0])
                    states[-1][1].append(possible_ends[min_idx][1])
                    print("Finalized end")
                    print(states[-1])
                    n_valid_samples += 1
                    valid_state_traces.append(states[-1])
                    break

        s = valid_state_traces[0]
        s0 = np.array(s[0])
        s1 = np.array(s[1])
        s2 = np.array(s[2])
        bot = s2 + mcts.state_manager.offset_value
        bot = bot[:len(s0)]
        mid = bot + s1
        top = bot + s0
        parts = [list(top), list(mid), list(bot)]
        durations = [['4'] * len(p) for p in parts]
        durations = [[int(di) for di in d] for d in durations]
        interval_figures = intervals_from_midi(parts, durations)
        _, interval_durations = fixup_parts_durations(parts, durations)
        all_parts.append(parts)
        all_durations.append(durations)
        print("completed {}".format(guide_idx))
    key_signature = "C"
    time_signature = "4/4"
    clefs = ["treble", "treble", "bass"]
    # now dump samples
    pitches_and_durations_to_pretty_midi(all_parts, all_durations,
                                         save_dir="three_voice_puct_mcts_samples",
                                         name_tag="three_voice_puct_mcts_sample_{}.mid",
                                         default_quarter_length=240,
                                         voice_params="piano")

    plot_pitches_and_durations(all_parts, all_durations,
                               save_dir="three_voice_puct_mcts_plots",
                               name_tag="three_voice_puct_mcts_plot_{}.ly",
                               #interval_figures=interval_figures,
                               #interval_durations=interval_durations,
                               use_clefs=clefs)

    # add caching here?
    # minimal check during rollout
    from IPython import embed; embed(); raise ValueError()
