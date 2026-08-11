from pathlib import Path

src = Path("/mnt/data/train_error_driven_baseline_v1_fixed2.py")
text = src.read_text(encoding="utf-8")

# Fix the regression-test bug:
# predict() returns (predictions, confidences), but the old code assigned
# the second value to p and then passed confidences to accuracy_score.
old = """t,p=predict([x[0] for x in REG]); r=accuracy_score([x[1] for x in REG],p); print('\\nTARGETED REGRESSION:',f'{r*100:.2f}%'); pd.DataFrame({'text':[x[0] for x in REG],'expected':[x[1] for x in REG],'predicted':p,'confidence':conf if False else [float(z) for z in predict([x[0] for x in REG])[1]]}).to_csv(OUT/'targeted_regression_results.csv',index=False)"""

new = """texts=[x[0] for x in REG]
expected=[x[1] for x in REG]
predicted, confidence=predict(texts)
r=accuracy_score(expected, predicted)
print('\\nTARGETED REGRESSION:', f'{r*100:.2f}%')
pd.DataFrame({
    'text': texts,
    'expected': expected,
    'predicted': predicted,
    'confidence': [float(z) for z in confidence],
    'correct': [a == b for a, b in zip(expected, predicted)]
}).to_csv(OUT/'targeted_regression_results.csv', index=False)"""

if old not in text:
    # More robust fallback: locate the line containing the broken regression call.
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines)
               if "TARGETED REGRESSION" in line and "accuracy_score" in line]
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely locate the broken regression-test line.")
    i = matches[0]
    replacement = new.splitlines()
    lines[i:i+1] = replacement
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
else:
    text = text.replace(old, new, 1)

out = Path("/mnt/data/train_error_driven_baseline_v1_fixed3.py")
out.write_text(text, encoding="utf-8")
print(out)
