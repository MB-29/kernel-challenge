from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

COLOUR_MAP = {
    'G': 'yellow',
    'A': 'green',
    'T': 'red',
    'C': 'blue',
}
DIGIT_MAP = {
    'G': 0,
    'A': 1,
    'T': 2,
    'C': 3,
}

def plot_sequence_colours(ax, sequence):
    n = len(sequence)
    for index, char in enumerate(sequence):
        ax.axis('off')
        ax.scatter(index, 0, color=COLOUR_MAP[char])


def plot_alignment(ax, source, target, indices):
    n = len(source)
    m = len(indices)
    source_x, target_x = np.arange(n), np.arange(n)

    for i in range(m):
        source_index, target_index = indices[i, 0], indices[i, 1]
        shift = source_x[source_index] - target_x[target_index]
        if shift >= 0:
            target_x[target_index:] += shift
        else:
            source_x[source_index:] -= shift

    ax.axis('off')
    ax.set_ylim((-2, 2))
    for i in range(n):
        source_char = source[i]
        target_char = target[i]
        ax.scatter(source_x[i], 1, color=COLOUR_MAP[source_char])
        ax.scatter(target_x[i], -1, color=COLOUR_MAP[target_char])
    for i in range(m):
        source_char, target_char = source[indices[i, 0]], target[indices[i, 1]]
        color = COLOUR_MAP[source_char] if source_char == target_char else 'black'
        ls = '-' if source_char == target_char else '--'
        source_index, target_index = indices[i, :]
        ax.plot((source_x[source_index], target_x[target_index]), (1, -1), color=color, ls=ls)



def plot_embedding(ax, embedding):
    ax.plot(embedding, color='black')



