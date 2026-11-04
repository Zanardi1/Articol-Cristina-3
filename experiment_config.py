"""Central configuration for the LFW facial-recognition experiment.

Defines the random seeds, train/test proportion, LFW loading parameters,
and expected dataset size and number of identities.

Imported by the experiment, reporting, and evaluation-audit modules.
Changing these values changes the experimental configuration.
"""

SEEDS = (42, 123, 2024, 2025, 2026)

TEST_SIZE = 0.2

MIN_FACES_PER_PERSON = 70

IMAGE_RESIZE = 0.4

EXPECTED_SAMPLES = 1288

EXPECTED_CLASSES = 7
