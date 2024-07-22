def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a character sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    Example
    -------
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return results
