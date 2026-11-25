"""AAMI three-class mapping for patient-independent N / S / V screening. F and Q beats are not classified (label -1)
but remain in the R-peak series, so RR intervals and context sequences are physiologically complete."""
CLASSES = ("N", "S", "V")
AAMI3 = {**{k: 0 for k in "NLRej"}, **{k: 1 for k in "AaJS"}, **{k: 2 for k in "VE"}}
BEAT_SYMBOLS = set("NLRejAaJSVEF/fQ")


def label(symbol: str) -> int:
    return AAMI3.get(symbol, -1)
