"""Entry point for the complete LFW facial-recognition experiment.

Run: python main.py
Delegates to multi_split.main(), which evaluates five pipelines over five
shared stratified train/test partitions and writes results to Results/.
Running this file can overwrite existing experimental outputs.
"""

from multi_split import main

if __name__ == '__main__':
    main()
