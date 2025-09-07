This is a work in progress

Recently, while commuting to work, I was driving in the straight-through lane with a car ahead of me and 
another in the left-turn-only lane. When the light turned green, 
the car in the turn-only lane suddenly cut across, 
forcing the car in front of me to brake hard. 
I slammed on my brakes as well but not quite in time, 
and I very lightly rear-ended the car ahead. 
Fortunately, neither of our cars were damaged. 
Still, I realized that if I had reacted just 100 milliseconds sooner, 
I likely would have avoided the collision entirely.

For context, the average human reaction time is around 220 milliseconds [source](). 
What if technology could cut that delay? A brain–computer interface, 
capable of detecting the neural signals that indicate the intention to brake, 
could activate emergency braking almost instantly—dramatically reducing reaction time 
and potentially preventing accidents.


According to Haufe et al., 2011:
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
