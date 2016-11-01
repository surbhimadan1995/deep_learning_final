"""
Steps to read in file with one review per line (i.e one document per line)

- Remove HTML markup
- Break list of docs into list of sentences and each sentence into list of words
- Map any numeric token in text to "NUMBER"
- Map any symbol token that is not .!? to "SYMBOL"
- Map any < 5 frequency word token to "UNKNOWN"
- We should have 29493 vocab words by the end of this cleaning


"""
