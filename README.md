This is a work in progress

# Intro

Recently, on my commute to work, I was driving in the straight-through lane behind another car, 
with a third vehicle in the left-turn-only lane. When the light turned green, 
the driver in the turn lane suddenly cut across traffic, forcing the car ahead of me to brake hard. 
I slammed on my brakes as well but not quite in time, and I very lightly tapped the car in front. 
Fortunately, neither vehicle was damaged. Still, I couldn’t shake the realization that if I had reacted just 100
milliseconds sooner, I probably would have avoided the collision altogether.

According to the [Human Benchmark reaction test](https://humanbenchmark.com/tests/reactiontime),
the median reaction time is 273 ms; my own five-trial median was 202 ms. 
While useful as a baseline, this test hardly reflects real driving conditions.
In the browser task, participants are highly focused, aiming for the fastest possible click during a short window of concentration. 
Driving, on the other hand, involves divided attention—listening to music or podcasts, talking, or even texting 
(please don't text and drive).
Moreover, in a vehicle, the required action is much slower than a simple mouse click.
Visual information must travel from the eyes to the brain, be processed and evaluated, 
and then converted into a motor command that moves from the brain to the leg, releasing the accelerator and pressing the brake.

Studies confirm this gap. In automobile simulators with emergency braking tasks,
[Haufe et al. (2011)](https://www.researchgate.net/publication/51530664_EEG_potentials_predict_upcoming_emergency_braking_during_simulated_driving)
reported a median reaction time of 665 ms, and [Nagler et al. (1973)](https://www.sciencedirect.com/science/article/pii/0300943273900411)
reported an average of 627 ms.
That’s roughly 400 ms slower than reaction times under ideal lab conditions. 
At highway speeds of 100 km/h (~62 mph), 400 ms corresponds to about 11 meters traveled before braking begins. 
Even at 32 km/h (~20 mph) — a typical speed when passing through an intersection — that delay translates to about 3.5 meters,
more than enough to prevent a minor collision like mine.

What if technology could eliminate this lag? A brain–computer interface (BCI), 
capable of detecting neural signals that indicate the intention to brake, 
could activate the brakes almost instantly—cutting precious milliseconds and potentially preventing accidents.

In Haufe et al. (2011), 18 healthy participants followed a virtual racing car at 100 km/h in a driving simulator.
They were instructed to remain within 20 meters of the lead vehicle, which executed randomized emergency braking maneuvers.
Throughout the task, EEG, EMG, and EOG data were recorded, creating an ideal dataset for exploring neural signatures of braking intent.

Building on that setup, my goals are to:
1. Develop an AI model that monitors EEG and predicts when a driver intends to apply emergency braking,
enabling integration with a vehicle’s braking system to reduce reaction delays.
2. Ensure generalization so the model works across new participants without extensive retraining.
3. Operate in pseudo-online mode at 250 Hz to meet real-time performance requirements.
4. Deploy on embedded hardware (ATMEGA2560), demonstrating feasibility for integration into real systems.

# Haufe Notes:
- A simulated assistance system using EEG and
EMG was found to detect emergency brakings 130 ms earlier than a system relying only on
pedal responses. At 100 km h−1 driving speed, this amounts to reducing the braking distance
by 3.66 m. (abstract)
- Traffic accidents rank third among the causes of death in the USA and area largely caused by human errors (1.)
- A recent development is hybrid approaches combining information
from vehicle/surround sensors and human behaviour. Basic
safety measures are adopted once external (radar or laser)
sensors indicate a potential upcoming crash. If further ‘panic’ 
activity at the brake pedal is detected, it is interpreted as the
driver’s confirmation of the criticality of the situation. This
allows the system to go into an emergency braking procedure
as soon as the brake pedal is touched by the driver, which saves
time. However, the brake pedal response is only the very last
event in the cascade of behavioural responses triggered during
an emergency braking situation
- Target situations were defined as those in which the
braking response was given no earlier than 300 ms and no
later than 1200 ms after the lead vehicle’s braking onset
- Non-targets were obtained by collecting all data blocks
(1500 ms duration, 500 ms equidistant offset) that were at
least 3000 ms apart from any stimulus.
- The distribution of pooled response times in target situations
was skewed with percentiles P5 = 505 ms, P25 = 595 ms,
P50 = 665 ms (median), P75 = 750 ms and P95 = 910 ms.
Collisions with the lead vehicle occurred in 17 ± 10% of the
critical situations
- Braking response times were defined base don teh first noticeable (above noise-level) braking pedal deflection 
after an induced braking manoeuvre
- react_emg (the react time given in data) is when the leg first starts to move, based on EMG, compared to ^
which is when pedal deflection begins.

# Data
- cnt: Contains the data
  - T:  np.
  - clab
  - file
  - fs: sampling rate
  - title
  - x: continuous multivariate data
- mnt: Defines electrode positions
- mrk: contains the breaking events

- D: dimensionality of data
- X: Number of samples


[Data](https://bnci-horizon-2020.eu/database/data-sets#:~:text=24.%20Emergency%20braking%20during%20simulated%20driving%20(002%2D2016))
[Paper](http://dx.doi.org/10.1088/1741-2560/8/5/056001)
