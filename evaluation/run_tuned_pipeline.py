import sys
import os

project_root = os.path.abspath('...')
sys.path.append(project_root)

from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi_tuned

pipe = FaithMultiPipeline(model="llama3")
text = "Fortune cookies originated in Italy"

label, full_conf = pipe.get_prediction_and_confidence(text)
expl = pipe.generate_explanation(text)
toks = pipe.tokenize_explanation(expl)
erased = [pipe.erase_token_and_get_confidence(text, t, label) for t in toks]
suff_conf = pipe.get_suff_confidence(toks, label)
attrib = pipe.get_attribution_tokens(text)

res = compute_faith_multi_tuned(
    full_conf, toks, erased, suff_conf, attrib, toks, toks
)
print(res)