# Qualitative audit and answer transitions

All 24 examples below use the frozen analysis rule: lexicographically first question ID in each of four transition categories for each D/E condition. These are descriptive cases, not a representative sample. No examples were selected for favorable outcomes. Full raw segments are retained in the local result ZIP; this public summary contains IDs and extracted answers only.

| Condition | Category | Question ID | Gold | C answer | Intervention answer | C / intervention reasoning tokens | Actual Wait |
|---|---|---|---|---|---|---:|---:|
| D256 | rescued | test/algebra/170.json | 11 | 21 | 11 | 8192 / 256 | 0 |
| D256 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 10 | 397 / 256 | 0 |
| D256 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 = 2x | 200 / 200 | 0 |
| D256 | unchanged_right | test/algebra/1035.json | 3 | 3 | 3 | 8192 / 256 | 0 |
| D512 | rescued | test/algebra/1457.json | -120 | 24 | 3 * 5 * -8 = -120 | 601 / 512 | 0 |
| D512 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 1 | 397 / 512 | 0 |
| D512 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 = 2x | 200 / 200 | 0 |
| D512 | unchanged_right | test/algebra/1035.json | 3 | 3 | 3 | 8192 / 512 | 0 |
| D1024 | rescued | test/algebra/1960.json | 9 | 3 | 9 | 101 / 354 | 0 |
| D1024 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 1 | 397 / 1024 | 0 |
| D1024 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 = 2x | 200 / 200 | 0 |
| D1024 | unchanged_right | test/algebra/109.json | 0 | 0 | 0 | 291 / 291 | 0 |
| D2048 | rescued | test/algebra/1960.json | 9 | 3 | 9 | 101 / 354 | 0 |
| D2048 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 1 | 397 / 2048 | 0 |
| D2048 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 = 2x | 200 / 200 | 0 |
| D2048 | unchanged_right | test/algebra/1035.json | 3 | 3 | 3 | 8192 / 623 | 0 |
| E1 | rescued | test/algebra/1275.json | \frac{\sqrt{3}}{3} | 1/3 | \frac{\sqrt{3}}{3} | 311 / 637 | 1 |
| E1 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 1 | 397 / 8192 | 0 |
| E1 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 | 200 / 314 | 1 |
| E1 | unchanged_right | test/algebra/1035.json | 3 | 3 | 3 | 8192 / 8192 | 0 |
| E2 | rescued | test/algebra/1339.json | 600 | 144 | 600 | 0 / 8191 | 1 |
| E2 | spoiled | test/algebra/1031.json | \frac{11}{2} | 5.5 | 1 | 397 / 8192 | 0 |
| E2 | unchanged_wrong | test/algebra/1004.json | 4 | 8 = 2x | 8 | 200 / 428 | 2 |
| E2 | unchanged_right | test/algebra/1035.json | 3 | 3 | 3 | 8192 / 8192 | 0 |

## Reading the cases

- D256, algebra/170: C lists the correct integer range, then repeatedly counts overlapping endpoints and answers 21. The cap answers 11. The trace supports repetition and counting drift as a failure pattern; it does not prove shorter reasoning always helps.
- D256, algebra/1031: C answers 5.5, equivalent to the gold 11/2; the capped condition answers 10. Early answer transition can also lose a correct result. Longer D512/D1024 caps spoil this ID too, and their first segment can differ from C.
- D1024, algebra/1960: C answers 3; D1024 answers 9. C used only 101 reasoning tokens while D1024 used 354, both below the cap. This is not evidence that truncation alone rescued the example: independent trajectories already differ.
- Algebra/1004 stays wrong in the selected cap cases: the extracted body is an equation, 8 = 2x, rather than the gold value 4. An explicit-answer extraction success does not establish a correct, fully resolved answer.
- E1/E2 categories include both rescued and spoiled answers and cases with no actual Wait. The policy is conditional on a stop, so a nominal E condition is not evidence that continuation occurred. See the actual Wait column and the aggregate trace diagnostic.

## Aggregate checks

E1 rescues 17 and spoils 24 answers; E2 rescues 19 and spoils 18. E1 injects one Wait on 264/500 outputs. E2 injects two on 208 outputs, one on 54, and none on 238. There are no intercepted learned reasoning role-marker stops in the saved full test; intercepted stops are model EOS tokens (151645). This implementation therefore tests an EOS-based continuation policy, not proof of the original learned delimiter behavior.

First reasoning segments match C exactly for 395/500 E1 and 393/500 E2 outputs. For cap conditions, prefix compatibility is checked because shorter output is expected. Differences before intervention are retained and reported; their numerical cause was not isolated. No outputs were regenerated or grades repaired after inspection.
