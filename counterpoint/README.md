# Counterpoint
Use PUCT MCTS to find harmony according to a cantus firmus. 

Rules and environment based on the book "The Study of Counterpoint", from Johann Joseph Fux's "Gradus Ad Parnassum", translated by Alfred Mann.

Repo with latest code can be found in:
[https://github.com/kastnerkyle/exploring_species_counterpoint](https://github.com/kastnerkyle/exploring_species_counterpoint)

![alt_text](https://github.com/rllabmcgill/final-project-mcts_for_decision_time_planning/blob/master/counterpoint/trace_0.png)

![alt_text](https://github.com/rllabmcgill/final-project-mcts_for_decision_time_planning/blob/master/counterpoint/trace_8.png)

See `trace_0.wav` for an example rendering, using Timidity. All .mid and .ly files are included in saved_*

For viewing .ly files, in Linux lilypond is usually used, the pdf output can be created using `lilypond file.ly`. .mid files require a midi player, in Linux timidity is a common choice. After installing timidity, the audio can be heard using `timidity file.mid`.

To run the code (tested in Python 2.7), do `python three_voice_pucts_mcts.py`. This will begin running the planning algorithm against each cantus firmus from the set of three voice, species 1 examples from the book. After completing all of these traces, it will write out lilypond (.ly) files and midi files (.mid) showing the results

## Authors:
Kyle Kastner | email: kastnerkyle@gmail.com | [@kastnerkyle](http://github.com/kastnerkyle/)
