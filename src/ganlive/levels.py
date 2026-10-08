"""The 8-bit level, the one unit every measurement in this project is quoted in, and the
thresholds measured in it. Torch-free, so the engine and the dials can share them."""

#: One 8-bit level is 1/127.5 of a generator's [-1, 1] output range.
LEVEL = 127.5

#: The most a rewrite that should be exact may move the picture, in 8-bit levels. Loose
#: enough for a driver that reassociates a sum, tight enough that a frozen picture fails.
EXACT_LEVELS = 0.5

#: A dial or a direction moving less than this many 8-bit levels at full travel is not a
#: control.
FLOOR_LEVELS = 1.0

#: How many times what a random direction of the same length moves, on the same latents, a
#: direction must move to be a control rather than a walk. Relative, because some
#: checkpoints move hard along every direction.
RANDOM_FLOOR = 2.0
