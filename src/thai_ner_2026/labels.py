"""Label definitions used by the LST20 NER task."""

LST20_NER_TAGS = [
    "O",
    "B_BRN", "B_DES", "B_DTM", "B_LOC", "B_MEA", "B_NUM", "B_ORG", "B_PER", "B_TRM", "B_TTL",
    "I_BRN", "I_DES", "I_DTM", "I_LOC", "I_MEA", "I_NUM", "I_ORG", "I_PER", "I_TRM", "I_TTL",
    "E_BRN", "E_DES", "E_DTM", "E_LOC", "E_MEA", "E_NUM", "E_ORG", "E_PER", "E_TRM", "E_TTL",
]

LABEL2ID = {label: index for index, label in enumerate(LST20_NER_TAGS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
