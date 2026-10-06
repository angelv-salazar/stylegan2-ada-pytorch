# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Lightweight Colab launcher for StyleGAN2-ADA.
# Adds a --tick option without modifying the original train.py CLI.

import sys

from training import training_loop
import train


def _extract_tick(argv):
    """Extract --tick from argv and remove it before handing off to train.py."""
    tick = 4
    cleaned = [argv[0]]
    i = 1

    while i < len(argv):
        arg = argv[i]

        if arg.startswith('--tick='):
            tick = int(arg.split('=', 1)[1])
        elif arg == '--tick':
            if i + 1 >= len(argv):
                raise SystemExit('--tick requires an integer value')
            tick = int(argv[i + 1])
            i += 1
        else:
            cleaned.append(arg)

        i += 1

    if tick < 1:
        raise SystemExit('--tick must be at least 1 kimg')

    return tick, cleaned


def main():
    tick, cleaned_argv = _extract_tick(sys.argv)
    sys.argv[:] = cleaned_argv

    original_training_loop = training_loop.training_loop

    def training_loop_with_tick(*args, **kwargs):
        kwargs.setdefault('kimg_per_tick', tick)
        return original_training_loop(*args, **kwargs)

    training_loop.training_loop = training_loop_with_tick
    train.main()


if __name__ == '__main__':
    main()
